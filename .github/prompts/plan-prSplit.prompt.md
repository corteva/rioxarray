# PR Split Strategy

## MR 1: Add Convention Option & Move CF to `_convention/cf.py` (Foundation)

**Purpose:** Infrastructure changes that all other MRs depend on. Framework only - no Zarr implementation.

**Files to include:**
| File | Changes |
|------|---------|
| rioxarray/enum.py | Add `Convention` enum with **only CF** (Zarr added in MR2) |
| rioxarray/_options.py | Add `CONVENTION` option with validator |
| rioxarray/_convention/__init__.py | Create package |
| rioxarray/_convention/cf.py | Extract existing CF logic from `rioxarray.py` |
| rioxarray/rioxarray.py | Refactor to use `cf.read_crs()`, `cf.read_transform()`, `cf.read_spatial_dimensions()` |
| test/unit/test_options.py | Test `CONVENTION` option |
| test/unit/test_convention_cf.py | Unit tests for CF module |

**What to exclude from this MR:**
- ❌ Don't add `Convention.Zarr` enum value (added in MR2)
- ❌ Don't export `Convention` from `__init__.py` (per reviewer comment)
- ❌ Don't modify `.pre-commit-config.yaml`
- ❌ Don't modify `pyproject.toml` dependencies/pins

**Address reviewer feedback:**
- Move `read_spatial_dimensions` CF logic into `cf.py` (alfredoahds' suggestion)
- Create docs/enum.rst for `Convention` docs instead of adding to rioxarray.rst

---

## MR 2: Reading Zarr CRS, Transform, Spatial Dimensions

**Purpose:** Add read-only support for Zarr conventions.

**Files to include:**
| File | Changes |
|------|---------|
| rioxarray/enum.py | Add `Convention.Zarr` enum value |
| rioxarray/_convention/zarr.py | Only the **read** functions: `read_crs()`, `read_transform()`, `read_spatial_dimensions()`, parsing utilities |
| rioxarray/rioxarray.py | Add Zarr fallback logic to `crs`, `transform`, dimension detection |
| test/integration/test_integration_zarr_conventions.py | **Only** read tests |

**Address reviewer feedback:**
- ✅ Both conventions should be attempted when reading regardless of setting (priority changes only)
- ✅ Use `CRS.from_user_input()` instead of explicit PROJJSON handling in `crs.py`
- ✅ Use functional test format, not class format
- ✅ Imports at top of file

**Exclude from this MR:**
- ❌ `write_crs()`, `write_transform()`, `write_conventions()`
- ❌ `calculate_spatial_bbox()`, `_write_spatial_metadata()`
- ❌ Convention declaration writing (`add_convention_declaration`)

---

## MR 3: Writing Zarr CRS

**Purpose:** Add `rio.write_crs(convention=Convention.Zarr)` support.

**Files to include:**
| File | Changes |
|------|---------|
| rioxarray/_convention/zarr.py | Add `write_crs()`, `format_proj_wkt2()`, `add_convention_declaration()` for proj: only |
| rioxarray/rioxarray.py | Add `convention` parameter to `write_crs()` |
| Tests for write_crs with Zarr convention |

---

## MR 4: Writing Zarr Transform

**Purpose:** Add `rio.write_transform(convention=Convention.Zarr)` support.

**Files to include:**
| File | Changes |
|------|---------|
| rioxarray/_convention/zarr.py | Add `write_transform()`, `format_spatial_transform()`, `add_convention_declaration()` for spatial: |
| rioxarray/rioxarray.py | Add `convention` parameter to `write_transform()` |
| Tests for write_transform with Zarr convention |

**Address reviewer feedback:**
- ✅ Use `rasterio.transform.array_bounds` for `calculate_spatial_bbox()`

---

## MR 5+: Other Zarr Convention Components (Future)

**Purpose:** Additional features after core read/write is merged.

Candidates:
- `write_conventions()` convenience function (reviewer explicitly said to exclude from this PR)
- `_write_spatial_metadata()` with bbox, registration, shape
- Documentation updates (conventions.ipynb, crs_management.ipynb)
- Zarr-specific methods documentation

---

## Specific Code Changes Per Reviewer Comment

| Comment | Action | MR |
|---------|--------|-----|
| Use `rasterio.transform.array_bounds` | Replace `calculate_spatial_bbox()` | MR 4 |
| Both conventions attempted on read | Modify fallback logic - always try both | MR 2 |
| Don't use class format for tests | Refactor tests to functional | MR 2+ |
| Imports at top of file | Fix test file imports | MR 2+ |
| PROJJSON → `CRS.from_user_input` | Remove explicit dict handling in `crs.py` | MR 2 |
| Remove `write_conventions()` | Defer to future MR | MR 5+ |
| Remove zarr-specific methods from docs | Don't add to crs_management.ipynb | All |
| Don't export Convention from `__init__` | Keep in `enum` namespace | MR 1 |
| Move Convention docs to enum.rst | Create new docs file | MR 1 |
| Don't remove dependency pin | Revert pyproject.toml changes | MR 1 |
| Don't modify .pre-commit-config.yaml | Revert | MR 1 |
| Don't modify dependencies | Revert pyproject.toml | MR 1 |
| Common interface for conventions | Consider abstract base / protocol | MR 1 (optional) |

---

## Suggested Branch Strategy

```
master
  └── feature/convention-option-cf-refactor  (MR 1)
        └── feature/zarr-read-support        (MR 2, based on MR 1)
              └── feature/zarr-write-crs     (MR 3, based on MR 2)
                    └── feature/zarr-write-transform (MR 4, based on MR 3)
```
