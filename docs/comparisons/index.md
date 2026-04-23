# Pipeline Comparisons

Comparisons of ASP with other open-source stereo photogrammetry pipelines, using the [SpaceNet UCSD WorldView-3 example](../examples/notebooks/worldview_spacenet_ucsd_stereo) as a common test case.

::::{grid} 1
:gutter: 3

:::{grid-item-card} SETSM
:link: setsm
:link-type: doc

Surface Extraction with TIN-based Search-space Minimization (Ohio State / PGC).
:::

:::{grid-item-card} CARS
:link: cars
:link-type: doc

CNES open-source stereo pipeline with Pandora dense matching.
:::

::::

## Key Differences

| Feature | ASP | CARS | SETSM |
|---|---|---|---|
| Language | C++ | Python + C++ (pybind11) | C++ |
| Dense matching | NCC, MGM, SGM, etc. | Pandora (Census + SGM) | TIN-based NCC |
| Bundle adjustment | Integrated (`bundle_adjust`) | Separate tool (optional) | None (RPC bias comp. only) |
| Memory management | In-process | Tiled, per-worker limits | Full image by default; sub-scene via `-boundary_*` flags |
| Parallelism | OpenMP threads | multiprocessing / Dask | OpenMP / MPI |
| Cropped processing | `mapproject` with crop window | ROI config or `cars-extractroi` | `-boundary_*` flags on full images |
| Install | Pre-built binaries | pip / Docker | Build from source |
| License | Apache 2.0 | Apache 2.0 | Apache 2.0 |

```{toctree}
:maxdepth: 1
:hidden:

setsm
cars
```
