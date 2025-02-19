from collections.abc import Hashable, Mapping
from typing import Any

import numpy as np
from affine import Affine
from xarray import DataArray, Index, Variable
# TODO: import from public API once it is available
from xarray.core.indexes import CoordinateTransformIndex, PandasIndex
from xarray.core.indexing import IndexSelResult, merge_sel_results
from xarray.core.coordinate_transform import CoordinateTransform


class AffineTransform(CoordinateTransform):
    """Affine 2D transform wrapper."""

    affine: Affine
    xy_dims: tuple[str, str]

    def __init__(
        self,
        affine: Affine,
        width: int,
        height: int,
        x_dim: str = "x",
        y_dim: str = "y",
        dtype: Any = np.dtype(np.float64),
    ):
        super().__init__((x_dim, y_dim), {x_dim: width, y_dim: height}, dtype=dtype)
        self.affine = affine

        # array dimensions in reverse order (y = rows, x = cols)
        self.xy_dims = self.dims[0], self.dims[1]
        self.dims = self.dims[1], self.dims[0]

    def forward(self, dim_positions):
        positions = tuple(dim_positions[dim] for dim in self.xy_dims)
        x_labels, y_labels = self.affine * positions

        results = {}
        for name, labels in zip(self.coord_names, [x_labels, y_labels]):
            results[name] = labels

        return results

    def reverse(self, coord_labels):
        labels = tuple(coord_labels[name] for name in self.coord_names)
        x_positions, y_positions = ~self.affine * labels

        results = {}
        for dim, positions in zip(self.xy_dims, [x_positions, y_positions]):
            results[dim] = positions

        return results

    def equals(self, other):
        if not isinstance(other, AffineTransform):
            return False
        return self.affine == other.affine and self.dim_size == other.dim_size


class AxisAffineTransform(CoordinateTransform):
    """Axis-independent wrapper of an affine 2D transform with no skew/rotation."""

    affine: Affine
    is_xaxis: bool
    coord_name: Hashable
    dim: str
    size: int

    def __init__(
        self,
        affine: Affine,
        size: int,
        dim: str,
        is_xaxis: bool,
        dtype: Any = np.dtype(np.float64),
    ):
        assert (affine.is_rectilinear and (affine.b == affine.d == 0))

        super().__init__((dim), {dim: size}, dtype=dtype)
        self.affine = affine
        self.is_xaxis = is_xaxis
        self.coord_name = dim
        self.dim = dim
        self.size = size

    def forward(self, dim_positions: dict[str, Any]) -> dict[Hashable, Any]:
        positions = dim_positions[self.dim]

        if self.is_xaxis:
            labels, _ = self.affine * (positions, np.zeros_like(positions))
        else:
            _, labels = self.affine * (np.zeros_like(positions), positions)

        return {self.coord_name: labels}

    def reverse(self, coord_labels: dict[Hashable, Any]) -> dict[str, Any]:
        labels = coord_labels[self.coord_name]

        if self.is_xaxis:
            positions, _ = ~self.affine * (labels, np.zeros_like(labels))
        else:
            _, positions = ~self.affine * (np.zeros_like(labels), labels)

        return {self.dim: positions}

    def equals(self, other):
        if not isinstance(other, AxisAffineTransform):
            return False

        # only compare the affine parameters of the relevant axis
        if self.is_xaxis:
            affine_match = self.affine.a == other.affine.a and self.affine.c == other.affine.c
        else:
            affine_match = self.affine.e == other.affine.e and self.affine.f == other.affine.f

        return affine_match and self.size == other.size

    def generate_coords(
        self, dims: tuple[str, ...] | None = None
    ) -> dict[Hashable, Any]:
        assert dims is None or dims == self.dims
        return self.forward({self.dim: np.arange(self.size)})

    def slice(self, slice: slice) -> "AxisAffineTransform":
        start = max(slice.start or 0, 0)
        stop = min(slice.stop or self.size, self.size)
        step = slice.step or 1

        # TODO: support reverse transform (i.e., start > stop)?
        assert slice.start < slice.stop

        size = stop - start // step
        scale = 1. / step

        if self.is_xaxis:
            affine = self.affine * Affine.translation(start, 0.) * Affine.scale(scale, 1.)
        else:
            affine = self.affine * Affine.translation(0., start) * Affine.scale(1., scale)

        return type(self)(affine, size, self.dim, is_xaxis=self.is_xaxis, dtype=self.dtype)


class AxisAffineTransformIndex(CoordinateTransformIndex):
    """Axis-independent Xarray Index for an affine 2D transform with no
    skew/rotation.

    For internal use only.

    This Index class provides specific behavior on top of
    Xarray's `CoordinateTransformIndex`:

    - Data slicing computes a new affine transform and returns a new
      `AxisAffineTransformIndex` object

    - Otherwise data selection creates and returns a new Xarray
      `PandasIndex` object for non-scalar indexers

    """
    axis_transform: AxisAffineTransform
    dim: str

    def __init__(self, transform: AxisAffineTransform):
        assert isinstance(transform, AxisAffineTransform)
        super().__init__(transform)
        self.axis_transform = transform
        self.dim = transform.dim

    def isel(  # type: ignore[override]
        self, indexers: Mapping[Any, int | slice | np.ndarray | Variable]
    ) -> "AxisAffineTransformIndex | PandasIndex | None":
        idxer = indexers[self.dim]

        # generate a new index with updated transform if a slice is given
        if isinstance(idxer, slice):
            return AxisAffineTransformIndex(self.axis_transform.slice(idxer))
        # no index for scalar value
        elif np.isscalar(idxer):
            return None
        # otherwise return a PandasIndex with values computed by forward transformation
        else:
            values = np.asarray(self.axis_transform.forward({self.dim: idxer}))
            return PandasIndex(values, self.dim, coord_dtype=values.dtype)

    def sel(self, labels, method=None, tolerance=None):
        coord_name = self.axis_transform.coord_name
        label = labels[coord_name]

        if isinstance(label, slice):
            if label.step is None:
                # continuous interval slice indexing (preserves the index)
                pos = self.transform.reverse({coord_name: np.array([label.start, label.stop])})
                pos = np.round(pos[self.dim]).astype("int")
                new_start = max(pos[0], 0)
                new_stop = min(pos[1], self.axis_transform.size)
                return IndexSelResult({self.dim: slice(new_start, new_stop)})
            else:
                # otherwise convert to basic (array) indexing
                label = np.arange(label.start, label.stop, label.step)

        # support basic indexing (in the 1D case basic vs. vectorized indexing
        # are pretty much similar)
        unwrap_xr = False
        if not isinstance(label, Variable | DataArray):
            # basic indexing -> either scalar or 1-d array
            try:
                var = Variable("_", label)
            except ValueError:
                var = Variable((), label)
            labels = {self.dim: var}
            unwrap_xr = True

        result = super().sel(labels, method=method, tolerance=tolerance)

        if unwrap_xr:
            dim_indexers = {self.dim: result.dim_indexers[self.dim].values}
            result = IndexSelResult(dim_indexers)

        return result


class RectilinearAffineTransformIndex(Index):
    """Xarray index for 2D rectilinear affine transform (no skew/rotation).

    For internal use only.

    """
    def __init__(
        self,
        x_index: AxisAffineTransformIndex,
        y_index: AxisAffineTransformIndex,
    ):
        self.x_index = x_index
        self.y_index = y_index

    def sel(
        self, labels: dict[Any, Any], method=None, tolerance=None
    ) -> IndexSelResult:
        results = []

        for axis_index in (self.x_index, self.y_index):
            coord_name = axis_index.axis_transform.coord_name
            if coord_name in labels:
                results.append(axis_index.sel({coord_name: labels[coord_name]}, method=method, tolerance=tolerance))

        return merge_sel_results(results)

    def equals(self, other: "RectilinearAffineTransformIndex") -> bool:
        return self.x_index.equals(other.x_index) and self.y_index.equals(other.y_index)


class RasterIndex(Index):
    """Xarray custom index for raster coordinates."""

    _x_index: AxisAffineTransformIndex | PandasIndex | None
    _y_index: AxisAffineTransformIndex | PandasIndex | None
    _xy_index: CoordinateTransformIndex | None

    def __init__(
        self,
        x_index: AxisAffineTransformIndex | PandasIndex | None = None,
        y_index: AxisAffineTransformIndex | PandasIndex | None = None,
        xy_index: CoordinateTransformIndex | None = None,
    ):
        # must at least have one index passed
        assert any(idx is not None for idx in (x_index, y_index, xy_index))
        # either 1D x/y coordinates with x_index/y_index or 2D x/y coordinates with xy_index
        if xy_index is not None:
            assert x_index is None and y_index is None

        self._x_index = x_index
        self._y_index = y_index
        self._xy_index = xy_index

    def _get_subindexes(self) -> tuple[Index | None, ...]:
        return (self._xy_index, self._x_index, self._y_index)

    @classmethod
    def from_transform(cls, affine: Affine, width: int, height: int, x_dim: str = "x", y_dim: str = "y") -> "RasterIndex":
        if affine.is_rectilinear and affine.b == affine.d == 0:
            x_transform = AxisAffineTransform(affine, width, x_dim, is_xaxis=True)
            y_transform = AxisAffineTransform(affine, height, y_dim, is_xaxis=False)
            return cls(
                x_index=AxisAffineTransformIndex(x_transform),
                y_index=AxisAffineTransformIndex(y_transform),
            )
        else:
            xy_transform = AffineTransform(affine, width, height, x_dim=x_dim, y_dim=y_dim)
            return cls(xy_index=CoordinateTransformIndex(xy_transform))

    @classmethod
    def from_variables(
        cls,
        variables: Mapping[Any, Variable],
        *,
        options: Mapping[str, Any],
    ) -> "RasterIndex":
        # TODO: compute bounds, resolution and affine transform from explicit coordinates.
        raise NotImplementedError(
            "Creating a RasterIndex from existing coordinates is not yet supported."
        )

    def create_variables(
        self, variables: Mapping[Any, Variable] | None = None
    ) -> dict[Hashable, Variable]:
        new_variables: dict[Hashable, Variable] = {}

        for index in (self._x_index, self._y_index, self._xy_index):
            if index is not None:
                new_variables.update(index.create_variables())

        return new_variables

    def isel(
        self, indexers: Mapping[Any, int | slice | np.ndarray | Variable]
    ) -> "RasterIndex | None":
        indexes: dict[str, Any] = {}

        if self._xy_index is not None:
            indexes["xy_index"] = self._xy_index.isel(indexers)

        if self._x_index is not None and self._x_index.dim in indexers:
            dim = self._x_index.dim
            indexes["x_index"] = self._x_index.isel(indexers={dim: indexers[dim]})

        if self._y_index is not None and self._y_index.dim in indexers:
            dim = self._y_index.dim
            indexes["x_index"] = self._y_index.isel(indexers={dim: indexers[dim]})

        if any(idx is not None for idx in indexes.values()):
            return RasterIndex(**indexes)
        else:
            return None

    def sel(
        self, labels: dict[Any, Any], method=None, tolerance=None
    ) -> IndexSelResult:
        results = []

        if self._xy_index is not None:
            results.append(self._xy_index.sel(labels, method=method, tolerance=tolerance))

        if self._x_index is not None and self._x_index.dim in labels:
            dim = self._x_index.dim
            results.append(self._x_index.sel(labels={dim: labels[dim]}, method=method, tolerance=tolerance))

        if self._y_index is not None and self._y_index.dim in labels:
            dim = self._y_index.dim
            results.append(self._y_index.sel(labels={dim: labels[dim]}, method=method, tolerance=tolerance))

        return merge_sel_results(results)

    def equals(self, other: "RasterIndex") -> bool:
        if not isinstance(other, RasterIndex):
            return False

        for (idx, oidx) in zip(self._get_subindexes(), other._get_subindexes()):
            if idx is not None and not idx.equals(oidx)
        if self._xy_index is not None and not self._xy_index.equals(other._xy_index):
            return False

        return self.x_index.equals(other.x_index) and self.y_index.equals(other.y_index)
