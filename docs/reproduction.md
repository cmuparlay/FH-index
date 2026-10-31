# Reproducing the FH-index experiments

This guide covers the implementation, the three analysis experiments, and the
paper's archived system evaluation. Use a new output directory for each run.
Keep source revisions, dataset checksums, commands, and environment records with
the results.

## 1. Choose the source and results you want to reproduce

| Goal | Source to use | Outputs |
|---|---|---|
| Check this release and its public API | This repository | Tests, examples, small generated-input checks |
| Rerun noise accumulation, PGM lookup-cost analysis, or partition counts using this release | `experiments/` in this repository | Newly computed analysis CSVs and plots |
| Regenerate the paper's system figures from its saved measurements | The frozen `fh-index-artifact` supplementary package | Derived tables and figures, without downloading large datasets |
| Rerun the paper's cross-index throughput, latency, sensitivity, and scalability experiments | The benchmark framework and dependency snapshots inside `fh-index-artifact` | New full-size system measurements |

The manuscript's [supplementary-material link](https://drive.google.com/drive/folders/1UqfPqiZg_PfBbrcoVy1ylqmtxyfLTwef?usp=sharing)
contains `initial_supplementary_materials` and
`revised_supplementary_materials (for VLDB 2027 revision)`. Use the revised
folder for this guide. Extract `fh-index-artifact.zip`
and identify the directory containing `code/`, `datasets/`, `results/`, and
`reproduce/`. In commands below, `ARTIFACT` means that extracted directory.
The source in that package is a historical snapshot. Replacing it with this
release creates a new experiment; record the substitution instead of labelling
the result an exact replay of the saved measurements.

## 2. Build and check this release

Use Linux x86-64, GCC 13.3 or another C++20 compiler, CMake 3.16 or newer,
and Python 3.12. The pinned Python environment was validated with Python 3.12.3.
The PGM timing experiment uses Clang 18.1.3 for the paper configuration.

```sh
git clone --recurse-submodules https://github.com/cmuparlay/FH-index.git
cd FH-index
python3.12 -m venv .venv
.venv/bin/python -m pip install -r experiments/requirements.txt
.venv/bin/python -m pip check
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=g++ -DFH_INDEX_BUILD_EXPERIMENTS=ON
cmake --build build -j4
ctest --test-dir build --output-on-failure
.venv/bin/python scripts/check_release.py --build-dir build
```

`check_release.py` is a repeatable local integration check, not a CI service.
It checks RO/RW examples, raw-key sorting and I/O failures, valid and truncated
SOSD inputs, PGM sentinel handling, the three analysis drivers and plots,
benchmark scripts, and consumption from
another CMake project. Inputs contain only 1,000 keys. It creates a fresh output
directory, records the command and output of every check, and writes
`verification.json`. Its timings are not paper performance results.

Use `--output-dir /path/to/new-check` to choose the output directory. The script
refuses to reuse an existing directory. `--cxx`, `--pgm-cxx`, `--threads`, and
`--jobs` select the compilers and resources. Install the pinned Python packages
before running it. No large datasets are required for this check.

For memory, ownership, and alignment checks, use a separate Debug build:

```sh
cmake -S . -B build-sanitize -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_CXX_COMPILER=g++ -DFH_INDEX_ENABLE_SANITIZERS=ON \
  -DFH_INDEX_BUILD_BENCHMARKS=OFF -DFH_INDEX_BUILD_UTILITIES=OFF \
  -DFH_INDEX_BUILD_EXPERIMENTS=OFF
cmake --build build-sanitize -j4
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \
UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
  ctest --test-dir build-sanitize --output-on-failure -j1
```

This includes lifecycle regressions for RO moves, RW repeated bulk loading,
destruction, and 16-byte-aligned values, plus boundary lookup/scan checks at
zero and `UINT64_MAX`. It does not measure performance.

## 3. Obtain and validate the paper datasets

The [dataset guide](../datasets/README.md) links the Kaggle distribution and
the [manifest](../datasets/manifest.json). The post-revision paper uses dataset
version 10: ten real datasets and four synthetic datasets with 200M keys each,
plus 800M-key Uniform, COVID, and OSM inputs.

Each analysis input is `[uint64 count][count uint64 keys]` in little-endian
order. The files are sorted. Check file names, byte sizes, and SHA-256 hashes
against the manifest before comparing results. For example:

```sh
sha256sum /path/to/data/books_200M_uint64.bin
```

The implementation-only programs in `benchmark/` expect **raw keys without
the count header**. Use the [explicit conversion procedure](benchmarks.md#dataset-format)
for those programs. The archived system framework has its own loaders and input
instructions; do not convert its inputs by guessing from a file name.

## 4. Rerun the three analysis experiments

Run the following from this repository root. Substitute the real input and new
output paths. Full-size runs consume substantially more memory and time than
the release check.

### Deviation accumulation

```sh
.venv/bin/python experiments/noise_accumulation/noise_accumulation.py \
  --data-dir /path/to/data --analysis 0 200M \
  --out-dir /path/to/new-run/noise --bins 100 --save-pdf --contact-sheet
```

Use `--datasets wise_200M_uint64.bin` for the WISE example in the paper, or omit
`--datasets` for the default eleven-input analysis population. The script fits
an OLS position model over the selected rank range and writes the summary,
window statistics, and figures. `--analysis 99M 101M` selects the central 2M-key
diagnostic instead of the full range. See the
[module protocol](../experiments/noise_accumulation/README.md).

### PGM lookup-cost decomposition

```sh
numactl --physcpubind=0 --membind=0 .venv/bin/python \
  experiments/pgm_statistics/pgm_statistics.py \
  --data-dir /path/to/data --analysis 0 200M \
  --out-dir /path/to/new-run/pgm --num-queries 100M --runs 3 \
  --seed 42 --warmup-queries 100M --epsilons 2 4 8 16 32 64 128 256 512 \
  --compiler clang++ --save-pdf --contact-sheet
```

The paper configuration pins this serial experiment to CPU 0 on NUMA node 0.
Queries are successful lookups sampled uniformly by rank. Each measured group
counterbalances partition-only and complete-lookup order. Last-mile time is
their paired difference. The script records the source hashes, PGM revision,
compiler, CPU affinity, and the number of excluded `UINT64_MAX` sentinel keys.
Use `--datasets covid_200M_uint64.bin` for the paper's COVID example. See the
[module protocol](../experiments/pgm_statistics/README.md).

### Partition counts

```sh
PARLAY_NUM_THREADS=192 numactl --cpunodebind=0 --membind=0 \
  .venv/bin/python experiments/partition_count/run_benchmark.py all \
  --data-dir /path/to/data --out-dir /path/to/new-run/partition-count \
  --compiler g++ --epsilon 64 --bucket-bound 128 --finger-sparsity 4
```

This uses GCC 13.3 for the paper configuration, PGM epsilon 64, and FH-index
`B=128`, `F=4` (target density `rho=1/4`, provisional construction density
`rho/2`). Both methods receive the same sorted, deduplicated keys after removing
the reserved `UINT64_MAX` sentinel. This release reads the SOSD header explicitly.
Compare integer partition counts before comparing rounded ratios or plots.

The default population is Uniform plus ten real datasets. `--resume` skips
completed input names in an existing result CSV; use a new output directory
when changing code, parameters, or input files. The corrected manuscript data,
archived original measurements, and release recheck are documented in
[partition-count results](../experiments/partition_count/reference/README.md).

## 5. Regenerate the paper's saved system figures

The complete saved measurements and plotting framework remain in the frozen
supplementary package. Use its own dependency file in a separate environment:

```sh
export ARTIFACT=/absolute/path/to/fh-index-artifact
cd "$ARTIFACT"
python3.12 -m venv .venv
.venv/bin/python -m pip install -r reproduce/requirements.txt
.venv/bin/python reproduce/artifact.py figures
.venv/bin/python reproduce/artifact.py compare
.venv/bin/python reproduce/artifact.py headline
```

The frozen artifact retains its original partition-count measurements. For the
corrected current manuscript figure, run `experiments/partition_count/plot_paper.py`
from this repository with
`experiments/partition_count/reference/paper_counts.csv`, as shown in the
[partition-count record](../experiments/partition_count/reference/README.md).

`figures` writes to `build/figures/`; `compare` checks the regenerated derived
tables against the frozen values; `headline` recomputes headline aggregates.
These commands do not launch the full benchmark workloads. Use a fresh
extraction if a previous `build/figures/` already exists: the artifact script
deliberately refuses to overwrite it.

## 6. Rerun the full system evaluation

The paper uses [Learned-Index-Test-Framework](https://github.com/xindubawukong/Learned-Index-Test-Framework),
developed by Hongbo Kang and Xiangyun Ding and based on
[GRE](https://github.com/gre4index/GRE). Credit the framework and cite GRE's
*Are Updatable Learned Indexes Ready?* (PVLDB 15(11), 3004–3017, 2022,
[DOI](https://doi.org/10.14778/3551793.3551848)); see
[BibTeX entries](references.bib). The local framework checkout inspected for
this release is commit `72322465be2e7c8779fd05cb337ec571b9336eee`.
The upstream framework URL may require repository access. For the paper's
archived measurements, use the frozen framework source and dependencies under
`code/benchmark/` in the public supplementary package described in Section 1;
do not assume that the current upstream commit is identical to that archive.

The reference machine was a dual-socket AMD EPYC 9654 system, restricted to NUMA
node 0: 96 physical cores and 192 hardware threads. System runs used GCC 13.3.0,
TBB-backed ParlayLib, jemalloc, and the recorded transparent-huge-page policy.
The two FH configurations are `B=32, rho=1` and the scan-oriented
`B=80, rho=1/3`. The complete framework also needs TBB, Intel MKL, jemalloc,
numactl, CMake, and the baseline dependency snapshots. Its original oneAPI
layout is `~/intel/oneapi/`; adapt installation paths explicitly if necessary.

Start with the package's correctness check and a dry run:

```sh
cd "$ARTIFACT"
cmake -S code/benchmark -B code/benchmark/build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=gcc-13 -DCMAKE_CXX_COMPILER=g++-13
cmake --build code/benchmark/build --target test_indexes_fh_index_rw_index_b32_f1
code/benchmark/build/test_indexes_fh_index_rw_index_b32_f1 \
  --num_keys=10000 --num_operations=20000
.venv/bin/python reproduce/run_experiments.py 1 \
  --data-dir /absolute/path/to/data --output-dir /absolute/path/to/new-system-run \
  --dry-run
```

Inspect the generated commands, CPU binding, dataset population, and allocator
configuration. Remove `--dry-run` to run on the intended benchmark machine.
Select experiment IDs independently:

| ID | Full system experiment |
|---|---|
| 1 | Read-write throughput, memory, and construction, including high mutation ratios |
| 2 | Bucket-size and finger-density sensitivity |
| 3 | Partition-finding and lookup timing |
| 4 | Read-only comparison |
| 5 | 800M-key scalability |
| 6 | Lookup/scan mixtures |
| 7 | Operation latency |
| 8 | Rebuild statistics |

Ordinary runs bulk-load half the dataset and execute one warmup and three
measured rounds of 200M operations. Exclude warmup measurements. Consult the
archived per-experiment commands for exceptions. The targeted FB/Genome rebuild
report uses `results/measurements/20260902_rebuild_kinds_fb_genome/run_rebuild_kinds.sh`;
experiment 8 is a broader suite. Concurrent rebuild event counts and durations
can differ between independent runs.

## 7. Map paper results to evidence

All paths in the right column are relative to `ARTIFACT`, except where explicitly
marked as paths in this repository. Match result labels rather than figure
numbers, which can change during typesetting.

| Paper result | Code and parameters | Saved measurements / derived output |
|---|---|---|
| WISE deviation accumulation | This repository: noise driver, full 200M range, 100 bins | `results/measurements/noise_accumulation/` |
| PGM lookup-cost curve on COVID | This repository: PGM driver, epsilon 2–512, serial counterbalanced protocol | `results/measurements/pgm_statistics/raw/` and `summary/` |
| PGM versus FH partition counts | This repository: epsilon 64, B 128, F 4 | Current figure: this repository's `experiments/partition_count/reference/paper_counts.csv`; legacy archive: `results/measurements/20260901_partition_count_final/` |
| Main read-write and read-only comparisons | Full system experiments 1 and 4 | `results/derived/01_main_results/` |
| Synthetic inputs and 800M scalability | Full system experiments 1 and 5 | `results/derived/02_scalability_and_robustness/` |
| Lookup/scan mixtures and mutation latency | Full system experiments 1, 6, and 7 | `results/derived/03_mixed_workloads_and_latency/` |
| B / rho sensitivity | Full system experiment 2 | `results/derived/04_parameter_sensitivity/` |
| Partition-finding timing and supplementary heatmaps | Full system experiment 3 | `results/derived/05_query_time_breakdown/` |
| Boundary propagation and rebuild observations | Saved alignment runs and targeted FB/Genome rebuild commands | `results/derived/06_construction_and_rebuild/` |

The artifact's `results/README.md` defines precedence among baseline reruns and
corrections; its `reproduce/FIGURE_MAP.md` lists the original source paths. Do
not pool superseded and final rows. Theoretical illustrations do not require
benchmark regeneration.

## 8. Interpret comparisons

- Compare matching datasets, workloads, thread counts, index parameters, and
  timing boundaries. Record deviations from the reference hardware/toolchain.
- Use raw values to compute geometric means, not rounded bars or speedups.
- Keep the dataset population complete for every reported aggregate.
- Treat small integration checks as execution checks, not performance evidence.
- Keep newly generated outputs separate from historical reference measurements.
- Cite the manuscript using [CITATION.cff](../CITATION.cff) or
  [BibTeX](references.bib); publication fields are omitted until finalized.
