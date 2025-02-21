import textwrap
from collections.abc import Hashable, Mapping
from typing import Any

import numpy as np
import pandas as pd
from affine import Affine
from xarray import DataArray, Index, Variable
from xarray.core.coordinate_transform import CoordinateTransform

# TODO: import from public API once it is available
from xarray.core.indexes import CoordinateTransformIndex, PandasIndex
from xarray.core.indexing import IndexSelResult, merge_sel_results


class AffineTransform(CoordinateTransform):
    """Affine 2D transform wrapper."""

    affine: Affine
    xy_dims: tuple[str, str]

    def __init__(
        self,
        affine: Affine,
        width: int,
        height: int,
        x_coord_name: Hashable = "xc",
        y_coord_name: Hashable = "yc",
        x_dim: str = "x",
        y_dim: str = "y",
        dtype: Any = np.dtype(np.float64),
    ):
        super().__init__(
            (x_coord_name, y_coord_name), {x_dim: width, y_dim: height}, dtype=dtype
        )
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
        coord_name: Hashable,
        dim: str,
        is_xaxis: bool,
        dtype: Any = np.dtype(np.float64),
    ):
        assert affine.is_rectilinear and (affine.b == affine.d == 0)

        super().__init__((coord_name,), {dim: size}, dtype=dtype)
        self.affine = affine
        self.is_xaxis = is_xaxis
        self.coord_name = coord_name
        self.dim = dim
        self.size = size

    def forward(self, dim_positions: dict[str, Any]) -> dict[Hashable, Any]:
        positions = np.asarray(dim_positions[self.dim])

        if self.is_xaxis:
            labels, _ = self.affine * (positions, np.zeros_like(positions))
        else:
            _, labels = self.affine * (np.zeros_like(positions), positions)

        return {self.coord_name: labels}

    def reverse(self, coord_labels: dict[Hashable, Any]) -> dict[str, Any]:
        labels = np.asarray(coord_labels[self.coord_name])

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
            affine_match = (
                self.affine.a == other.affine.a and self.affine.c == other.affine.c
            )
        else:
            affine_match = (
                self.affine.e == other.affine.e and self.affine.f == other.affine.f
            )

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
        assert start < stop

        size = (stop - start) // step
        scale = float(step)

        if self.is_xaxis:
            affine = (
                self.affine * Affine.translation(start, 0.0) * Affine.scale(scale, 1.0)
            )
        else:
            affine = (
                self.affine * Affine.translation(0.0, start) * Affine.scale(1.0, scale)
            )

        return type(self)(
            affine,
            size,
            self.coord_name,
            self.dim,
            is_xaxis=self.is_xaxis,
            dtype=self.dtype,
        )


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

    - The index can be converted to a `pandas.Index` object (useful for Xarray
      operations that don't work with Xarray indexes yet).

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
        # no index for vectorized (fancy) indexing with n-dimensional Variable
        elif isinstance(idxer, Variable) and idxer.ndim > 1:
            return None
        # no index for scalar value
        elif np.ndim(idxer) == 0:
            return None
        # otherwise return a PandasIndex with values computed by forward transformation
        else:
            values = self.axis_transform.forward({self.dim: idxer})[
                self.axis_transform.coord_name
            ]
            if isinstance(idxer, Variable):
                new_dim = idxer.dims[0]
            else:
                new_dim = self.dim
            return PandasIndex(values, new_dim, coord_dtype=values.dtype)

    def sel(self, labels, method=None, tolerance=None):
        coord_name = self.axis_transform.coord_name
        label = labels[coord_name]

        if isinstance(label, slice):
            if label.step is None:
                # continuous interval slice indexing (preserves the index)
                pos = self.transform.reverse(
                    {coord_name: np.array([label.start, label.stop])}
                )
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

    def to_pandas_index(self) -> pd.Index:
        values = self.transform.generate_coords()
        return pd.Index(values[self.dim])


# The types of Xarray indexes that may be wrapped by RasterIndex
WrappedIndex = AxisAffineTransformIndex | PandasIndex | CoordinateTransformIndex
WrappedIndexCoords = Hashable | tuple[Hashable, Hashable]


def _filter_dim_indexers(index: WrappedIndex, indexers: Mapping) -> Mapping:
    if isinstance(index, CoordinateTransformIndex):
        dims = index.transform.dims
    else:
        # PandasIndex
        dims = (str(index.dim),)

    return {dim: indexers[dim] for dim in dims if dim in indexers}


class RasterIndex(Index):
    """Xarray index for raster coordinates.

    RasterIndex is itself a wrapper around one or more Xarray indexes associated
    with either the raster x or y axis coordinate or both, depending on the
    affine transformation and prior data selection (if any):

    - The affine transformation is not rectilinear or has rotation: this index
      encapsulates a single `CoordinateTransformIndex` object for both the x and
      y axis (2-dimensional) coordinates.

    - The affine transformation is rectilinear ands has no rotation: this index
      encapsulates one or two index objects for either the x or y axis or both
      (1-dimensional) coordinates. The index type is either a subclass of
      `CoordinateTransformIndex` that supports slicing or `PandasIndex` (e.g.,
      after data selection at arbitrary locations).

    """

    _wrapped_indexes: dict[WrappedIndexCoords, WrappedIndex]

    def __init__(self, indexes: Mapping[WrappedIndexCoords, WrappedIndex]):
        idx_keys = list(indexes)
        idx_vals = list(indexes.values())

        # either one or the other configuration (dependent vs. independent x/y axes)
        axis_dependent = (
            len(indexes) == 1
            and isinstance(idx_keys[0], tuple)
            and isinstance(idx_vals[0], CoordinateTransformIndex)
        )
        axis_independent = len(indexes) in (1, 2) and all(
            isinstance(idx, AxisAffineTransformIndex | PandasIndex) for idx in idx_vals
        )
        assert axis_dependent ^ axis_independent

        self._wrapped_indexes = dict(indexes)

    @classmethod
    def from_transform(
        cls, affine: Affine, width: int, height: int, x_dim: str = "x", y_dim: str = "y"
    ) -> "RasterIndex":
        indexes: dict[
            WrappedIndexCoords, AxisAffineTransformIndex | CoordinateTransformIndex
        ]

        # pixel centered coordinates
        affine = affine * Affine.translation(0.5, 0.5)

        if affine.is_rectilinear and affine.b == affine.d == 0:
            x_transform = AxisAffineTransform(affine, width, "x", x_dim, is_xaxis=True)
            y_transform = AxisAffineTransform(
                affine, height, "y", y_dim, is_xaxis=False
            )
            indexes = {
                "x": AxisAffineTransformIndex(x_transform),
                "y": AxisAffineTransformIndex(y_transform),
            }
        else:
            xy_transform = AffineTransform(
                affine, width, height, x_dim=x_dim, y_dim=y_dim
            )
            indexes = {("x", "y"): CoordinateTransformIndex(xy_transform)}

        return cls(indexes)

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

        for index in self._wrapped_indexes.values():
            new_variables.update(index.create_variables())

        return new_variables

    def isel(
        self, indexers: Mapping[Any, int | slice | np.ndarray | Variable]
    ) -> "RasterIndex | None":
        new_indexes: dict[WrappedIndexCoords, WrappedIndex] = {}

        for coord_names, index in self._wrapped_indexes.items():
            index_indexers = _filter_dim_indexers(index, indexers)
            if not index_indexers:
                # no selection to perform: simply propagate the index
                # TODO: uncomment when https://github.com/pydata/xarray/issues/10063 is fixed
                # new_indexes[coord_names] = index
                ...
            else:
                new_index = index.isel(index_indexers)
                if new_index is not None:
                    new_indexes[coord_names] = new_index

        if new_indexes:
            # TODO: if there's only a single PandasIndex can we just return it?
            # (maybe better to keep it wrapped if we plan to later make RasterIndex CRS-aware)
            return RasterIndex(new_indexes)
        else:
            return None

    def sel(
        self, labels: dict[Any, Any], method=None, tolerance=None
    ) -> IndexSelResult:
        results = []

        for coord_names, index in self._wrapped_indexes.items():
            if not isinstance(coord_names, tuple):
                coord_names = (coord_names,)
            index_labels = {k: v for k, v in labels if k in coord_names}
            if index_labels:
                results.append(
                    index.sel(index_labels, method=method, tolerance=tolerance)
                )

        return merge_sel_results(results)

    def equals(self, other: Index) -> bool:
        if not isinstance(other, RasterIndex):
            return False
        if set(self._wrapped_indexes) != set(other._wrapped_indexes):
            return False

        return all(
            index.equals(other._wrapped_indexes[k])  # type: ignore[arg-type]
            for k, index in self._wrapped_indexes.items()
        )

    def to_pandas_index(self) -> pd.Index:
        # conversion is possible only if this raster index encapsulates
        # exactly one AxisAffineTransformIndex or a PandasIndex associated
        # to either the x or y axis (1-dimensional) coordinate.
        if len(self._wrapped_indexes) == 1:
            index = next(iter(self._wrapped_indexes.values()))
            if isinstance(index, AxisAffineTransformIndex | PandasIndex):
                return index.to_pandas_index()

        raise ValueError("Cannot convert RasterIndex to pandas.Index")

    def __repr__(self):
        items: list[str] = []

        for coord_names, index in self._wrapped_indexes.items():
            items += [repr(coord_names) + ":", textwrap.indent(repr(index), "    ")]

        return "RasterIndex\n" + "\n".join(items)
