# Paper analysis experiments

These are the three analysis modules from `fh_index_materials`, integrated
with this checkout. They cover the paper's modeling analyses; the separate
cross-index throughput framework is not included here. Generated measurements
are written to the requested output directory, not shipped as new paper results.

| Module | Question |
|---|---|
| [noise_accumulation](noise_accumulation/README.md) | How do local count deviations accumulate into global OLS rank error? |
| [pgm_statistics](pgm_statistics/README.md) | How does PGM epsilon affect partition finding, last-mile search, and memory? |
| [partition_count](partition_count) | How many partitions do PGM and FH construction produce at comparable search spans? |

## Requirements

Use Linux x86-64, Python 3.12, CMake 3.16 or newer, GCC 13 or Clang 18,
and the initialized submodules from the repository root. Python packages are
needed for noise analysis and plots; the PGM benchmark-only driver uses the
standard library.

`requirements.txt` pins the complete validated plotting environment. The
paper's frozen supplementary package retains its own separate environment.
See the [complete reproduction guide](../docs/reproduction.md) for both paths.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r experiments/requirements.txt
git submodule update --init --recursive
```

Run commands below from the repository root. `datasets/` is the default input
directory; use `--data-dir /path/to/data` to choose another location. The input
format and download source are in [datasets/README.md](../datasets/README.md).

## Noise accumulation

Full-range analysis, with a global OLS model fitted over the selected rank range:

```sh
.venv/bin/python experiments/noise_accumulation/noise_accumulation.py \
  --data-dir /path/to/data --analysis 0 200M \
  --out-dir results/noise_full --bins 100 --save-pdf --contact-sheet
```

Select `--analysis 99M 101M` for the central two-million-key window.
`--datasets filename.bin` restricts the input list. See the module README for
CSV fields and plotting behavior.

## PGM lookup-cost analysis

```sh
numactl --physcpubind=0 --membind=0 .venv/bin/python \
  experiments/pgm_statistics/pgm_statistics.py \
  --data-dir /path/to/data --analysis 0 200M \
  --out-dir results/pgm --num-queries 100M --runs 3 --seed 42 \
  --warmup-queries 100M --epsilons 2 4 8 16 32 64 128 256 512 \
  --compiler clang++ --save-pdf --contact-sheet
```

For the paper configuration use Clang 18.1.3, one pinned CPU on NUMA node 0.
The script builds its C++ backend and records the compiler and PGM revision.
Each measured group counterbalances partition-only and complete lookup order;
last-mile time is their paired difference. See the module README for details.

## Partition-count comparison

```sh
PARLAY_NUM_THREADS=4 .venv/bin/python experiments/partition_count/run_benchmark.py all \
  --data-dir /path/to/data --out-dir results/partition_count \
  --compiler g++ --epsilon 64 --bucket-bound 128 --finger-sparsity 4
```

The current manuscript figure uses the corrected
[paper counts](partition_count/reference/paper_counts.csv); see the
[correction and historical comparison](partition_count/reference/README.md).

The FH source defaults to this repository; an extra FH-index checkout is not
needed. `F=4` means target finger density `1/4`; construction uses provisional
density `1/8` before trimming. The comparison removes duplicates and the reserved
`UINT64_MAX` sentinel from both inputs. Files are read as SOSD data with an
8-byte count header; `--load-limit` limits keys after the header.

A small generated-input smoke run requires no downloaded dataset:

```sh
PARLAY_NUM_THREADS=4 .venv/bin/python experiments/partition_count/run_benchmark.py bench \
  --datasets GEN --load-limit 10000 --out-dir results/partition_smoke
```

To build both C++ backends in the root build, enable
`-DFH_INDEX_BUILD_EXPERIMENTS=ON`. To build only the partition backend:

```sh
cmake -S experiments/partition_count -B build-partition -DCMAKE_BUILD_TYPE=Release
cmake --build build-partition -j4
```

## Measurement scope

The original paper environment used a dual-socket AMD EPYC 9654 server, with
measurements restricted to NUMA node 0. Small smoke runs check execution and
input/output handling; their timings are not replacements for full-size paper
measurements. Record compiler versions, CPU binding, worker count, input
checksums, and commands for comparisons. Use separate output directories for
independent runs.

Sources were imported from `fh_index_materials` revision
`3c977031e5d2b8bb743acc11076137341d075448`; see
[third-party and source notices](../THIRD_PARTY_NOTICES.md).
