# FH-index

FH-index is a finger-hinted learned index for `uint64_t` keys. This is the source
repository accompanying **FH-index: A Finger-Hinted Learned Index**, with read-only
(RO) and concurrent read-write (RW) implementations and three paper analysis
experiments.

## Get the source and build

Requires Linux on x86-64, a C++20 compiler, CMake 3.16 or newer, and Git. The
release checks use GCC 13.3.0 and CMake 3.30.0. Clone with submodules; a GitHub
source ZIP does not include the ParlayLib or PGM-index sources.

```sh
git clone --recurse-submodules https://github.com/cmuparlay/FH-index.git
cd FH-index
# For an existing checkout:
git submodule update --init --recursive
cmake -S . -B build -DCMAKE_CXX_COMPILER=g++ -DCMAKE_BUILD_TYPE=Release
cmake --build build -j4
ctest --test-dir build --output-on-failure
PARLAY_NUM_THREADS=4 ./build/example_ro
PARLAY_NUM_THREADS=4 ./build/example_rw
```

Use `Debug` for development. `FH_INDEX_CPU` defaults to `native`; `baseline`,
`avx2`, and `avx512` select other CPU targets for project executables. Binaries
must run on a CPU supporting the selected instructions.

## Contents

| Path | Purpose |
|---|---|
| [include/fh_index](include/fh_index) | Public header-only RO and RW indexes |
| [examples](examples) | Small complete RO and RW programs |
| [test](test) | Correctness and concurrency tests |
| [benchmark](benchmark), [scripts](scripts) | Implementation benchmarks; [usage and input format](docs/benchmarks.md) |
| [docs/reproduction.md](docs/reproduction.md) | Complete reproduction guide and paper-result mapping |
| [experiments](experiments/README.md) | Noise accumulation, PGM lookup-cost analysis, and partition-count comparison |
| [datasets](datasets/README.md) | Paper dataset source, file format, and checksums; data are downloaded separately |
| [third_party](THIRD_PARTY_NOTICES.md) | Fixed dependencies, their origins, and licenses |

The three analysis experiments were integrated from `fh_index_materials`.
The paper's cross-index system evaluation uses
[Learned-Index-Test-Framework](https://github.com/xindubawukong/Learned-Index-Test-Framework),
developed by Hongbo Kang and Xiangyun Ding and based on
[GRE](https://github.com/gre4index/GRE). The framework and historical measurements
are distributed in the separate frozen supplementary package; see the
[reproduction guide](docs/reproduction.md#6-rerun-the-full-system-evaluation).

## Partition-count data correction

The manuscript figure now uses the corrected SOSD-reader results from the
2026-11-01 full-size recheck. PGM partition counts are 42,695 for COVID and
27,553 for WISE; all eleven FH counts are unchanged. The canonical figure
input is [paper_counts.csv](experiments/partition_count/reference/paper_counts.csv).
The [recheck record](experiments/partition_count/reference/README.md) preserves
the original measurements and the controlled reader comparison separately.

## Index usage and contracts

- [RO example](examples/ro.cpp): bulk loading, lookup, and scans.
- [RW example](examples/rw.cpp): insertion, update, deletion, lookup, and scans.

`BulkLoad` takes non-empty, sorted, unique keys and a matching value array.
Build before querying. Do not run RW `BulkLoad`, destruction, or object-lifetime
changes concurrently with operations on the same index. RO `BulkLoad` rejects
empty input with `std::invalid_argument` and preserves previously loaded data.
RW values must be trivially copyable, with alignment no greater than
`alignof(std::max_align_t)` (16 bytes on the supported x86-64 platform).
RO indexes support move construction and move assignment; moved-from indexes
must be bulk-loaded again before querying. Indexes cannot be copied.

`Find` reports whether a key was found. RW `Insert` returns false for an existing
key; `Update` and `Delete` return false for an absent key. `Scan(lower, count,
results)` writes at most `count` entries in ascending order, starting at keys
at least `lower`; the caller supplies space for `count` entries. During concurrent
updates, RW scans validate each bucket separately. Do not assume that a complete
scan represents one atomic snapshot across all buckets.

The RW epoch manager has `parlay::num_workers()` registration slots, shared
across RW index instances. Each OS thread that uses a guarded operation retains
one slot until that thread exits, even while idle. Include the calling/main
thread, Parlay workers, and any external thread-pool threads when setting
`PARLAY_NUM_THREADS` before process startup. Exceeding this registered-thread
capacity calls `abort()`; unrestricted external thread pools are not supported.

The example parameters label the constructor arguments. `B` is the target
bucket bound and `F` is keys per finger (density `1/F`); RW also takes overflow
bounds and separate internal-level settings. The top-level threshold controls
when construction stops adding levels. Use the examples' settings to start.

### Rebuild statistics

Instantiate `FHIndexRW<Key, Value, true>` to enable statistics.
`GetRetrainStats()` returns aggregate `count`, `total_ns`, and `max_ns`, plus
`by_kind[static_cast<size_t>(RebuildKind::kSubtree)]` and corresponding
`kTopLevelSubtree` and `kRoot` entries. The enum is in namespace `fh_index_rw`.
Read or reset counters after workers finish for coherent totals. `total_ns`
sums individual rebuild durations, which can overlap across threads; it is
not elapsed wall time. Statistics are disabled by default.

## Use in another CMake project

```cmake
add_subdirectory(path/to/FH-index fh_index_build)
target_link_libraries(your_target PRIVATE fh_index::fh_index)
```

The target supplies C++20, include paths, and thread linkage. Include
`fh_index/fh_index_ro.h` or `fh_index/fh_index_rw.h`; the class namespaces are
`fh_index_ro` and `fh_index_rw`. Consumers choose their own CPU flags.
When added as a subdirectory, FH-index's executables are disabled by default.

| CMake option | Standalone default | Purpose |
|---|---|---|
| `FH_INDEX_BUILD_TESTS` | `ON` | Correctness tests |
| `FH_INDEX_BUILD_EXAMPLES` | `ON` | RO/RW examples |
| `FH_INDEX_BUILD_BENCHMARKS` | `ON` | Implementation benchmarks |
| `FH_INDEX_BUILD_UTILITIES` | `ON` | Dataset utilities |
| `FH_INDEX_BUILD_EXPERIMENTS` | `OFF` | C++ backends for the paper analyses |
| `FH_INDEX_ENABLE_SANITIZERS` | `OFF` | ASan/UBSan on project executables |
| `FH_INDEX_ALLOCATOR` | `system` | Benchmark allocator; optionally `jemalloc` |
| `FH_INDEX_RECLAMATION` | `normal` | Benchmark memory reclamation; diagnostic modes described below |

`normal` enables epoch guards, retirement, and reclamation. The alternatives
`disabled`, `guards_only`, and `record_only` isolate those costs for diagnostic
experiments and disable automatic reclamation of replaced allocations. They
change the meaning of throughput and memory measurements; label them explicitly when
reporting results. Tests always use normal reclamation. See the
[mode descriptions](docs/benchmarks.md#reclamation-modes).

## License and citation

FH-index and its analysis drivers are [MIT licensed](LICENSE), copyright
Hongbo Kang and Xiangyun Ding. Dependencies retain their own licenses, including
Apache-2.0 for PGM-index; see [third-party notices](THIRD_PARTY_NOTICES.md).

Citation metadata is available in [CITATION.cff](CITATION.cff) and
[BibTeX](docs/references.bib). If using this work, cite the accompanying manuscript, *FH-index: A Finger-Hinted
Learned Index*, by Hongbo Kang, Xiangyun Ding, Yiwei Zhao, Yingdi Shan, Guy E.
Blelloch, Laxman Dhulipala, Yan Gu, Charles McGuffey, Mingxing Zhang, Yongwei Wu,
and Phillip B. Gibbons. Publication metadata will be added when finalized.

When using the cross-index evaluation framework, also credit
**Learned-Index-Test-Framework** and cite its upstream GRE paper:
Chaichon Wongkham et al., *Are Updatable Learned Indexes Ready?*, PVLDB 15(11),
3004–3017, 2022, [doi:10.14778/3551793.3551848](https://doi.org/10.14778/3551793.3551848).
Both entries are included in [the BibTeX file](docs/references.bib).
