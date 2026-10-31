# Third-party sources and licenses

FH-index code and the imported analysis drivers are distributed under the
[MIT license](LICENSE), copyright Hongbo Kang and Xiangyun Ding. Dependencies
retain the separate licenses and notices described below. Do not remove their
license files or embedded copyright notices when redistributing source or binaries.

| Component | Source revision | Use | License and distribution |
|---|---|---|---|
| [ParlayLib](https://github.com/cmuparlay/parlaylib) | `e1f1dc0ccf930492a2723f7fbef8510d35bf57f5` | Parallel primitives used by FH-index | [MIT](third_party/parlaylib/LICENSE); pinned Git submodule |
| [timer_tree](https://github.com/Loremkang/timer_tree) | `77079357999c760cb77c30b9d7a5ecc0b6054f1a` | Timing utilities for tests and benchmarks | [MIT](third_party/timer_tree/LICENSE); vendored headers |
| [parallel-minimum-interval-covering](https://github.com/Loremkang/parallel-minimum-interval-covering) | `5cba8e69a1926cf09e36cddb68048975053e1e97` | FH-index construction | [MIT](third_party/parallel-minimum-interval-covering/LICENSE); vendored headers |
| [PGM-index](https://github.com/gvinciguerra/PGM-index) | `c6fcf3d34e55eb0061b01e2f49dfcbdb711f1407` | Paper analysis experiments only | [Apache-2.0](third_party/PGM-index/LICENSE); pinned Git submodule |

ParlayLib is copyright Guy Blelloch and other contributors. Its
`include/parlay/internal/atomic_wait.h` separately retains the full MIT notice
for copyright 2019 NVIDIA Corporation. PGM-index retains its Apache-2.0 license
and Giorgio Vinciguerra's source-file copyright notices. Neither dependency
is relicensed by the FH-index root LICENSE.

The timer and interval-covering headers are copied without code changes from
the revisions above. Their author, Hongbo Kang, releases these copies under
the accompanying MIT licenses. Upstream build scripts, development notes, and
the interval-covering repository's separate Parlay submodule are not included;
the vendored headers use FH-index's pinned ParlayLib.

The three analysis modules under `experiments/` were imported from
[fh_index_materials](https://github.com/Loremkang/fh_index_materials), revision
`3c977031e5d2b8bb743acc11076137341d075448`. Local adaptations update the layout,
build integration, and dataset handling; the original repository history and
generated results are not part of this source snapshot.

Optional tools installed separately include NumPy, pandas, Matplotlib,
numactl, and jemalloc. These are not bundled or required by the header-only
target; their installed distributions supply their own license notices.

The separate paper-evaluation package uses
[Learned-Index-Test-Framework](https://github.com/xindubawukong/Learned-Index-Test-Framework)
by Hongbo Kang and Xiangyun Ding, based on [GRE](https://github.com/gre4index/GRE).
Neither framework is bundled in this checkout or relicensed by its root MIT
license. Framework attribution and the GRE paper citation are provided in
[the reproduction guide](docs/reproduction.md#6-rerun-the-full-system-evaluation)
and [BibTeX references](docs/references.bib).

When reporting experiments involving PGM-index, also cite:

> Paolo Ferragina and Giorgio Vinciguerra. The PGM-index: a fully-dynamic
> compressed learned index with provable worst-case bounds. PVLDB 13(8),
> 1162–1175, 2020. https://doi.org/10.14778/3389133.3389135
