# Implementation benchmarks

These programs exercise this FH-index implementation. The three separate
paper analyses are documented in [experiments/](../experiments/README.md).

## Build and run

The default root CMake build enables benchmarks. To build them explicitly:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DFH_INDEX_BUILD_BENCHMARKS=ON
cmake --build build -j4
PARLAY_NUM_THREADS=4 ./scripts/run_benchmarks.sh GEN
```

This runs RO and RW lookups for B in 32/40/48 and F in 2/4, using 100,000
generated keys and queries and three rounds. Results are small smoke workloads,
not the paper's full-system measurements. `run_benchmarks_ro.sh` restricts the
same driver to RO. Pass a dataset path instead of `GEN` to use real keys.

Both scripts accept these environment variables:

| Variable | Default | Meaning |
|---|---|---|
| `BUILD_DIR` | `build` in the checkout | Directory containing built executables |
| `RESULTS_DIR` | Timestamped directory under `results/` | Output logs |
| `PARLAY_NUM_THREADS` | `4` | Number of Parlay workers |
| `TYPES` | `RO RW` | Space-separated index versions |
| `BLOCK_SIZES` | `32 40 48` | B variants already built by CMake |
| `FINGER_SPARSITIES` | `2 4` | F variants already built by CMake |
| `LOAD_LIMIT` / `NUM_QUERIES` | `100000` | Keys to load / operations per round |
| `NUM_ROUNDS` / `SEED` | `3` / `42` | Rounds / dataset seed |
| `NUMA_NODE` | unset | Optional CPU and memory binding with numactl |

For example, a larger RO run with binding enabled explicitly:

```sh
NUMA_NODE=0 PARLAY_NUM_THREADS=192 LOAD_LIMIT=200000000 \
NUM_QUERIES=200000000 NUM_ROUNDS=4 \
./scripts/run_benchmarks_ro.sh /path/to/raw_keys.bin
```

The direct executable interface is:

```text
bench_fh_index_lat0_typeRO_b40_f2 dataset load_limit num_round num_queries granularity zipf_alpha scramble [seed] [bench_mode]
```

`bench_mode` is `find` (default), or `insert`/`delete` for RW. Granularity,
Zipf parameter, and scrambling are zero in the scripts above.

## Dataset format

Core benchmark programs and dataset utilities read a contiguous array of
little-endian `uint64_t` keys **without a count header**. The paper analyses in
`experiments/` instead read SOSD files **with an eight-byte count header**.
Raw readers reject files whose length is not a multiple of eight bytes and
report read failures. Writers report failed output operations.
Do not pass a headered file to a core benchmark. To convert a file without
loading it all into RAM:

```sh
python3 - /path/to/sosd.bin /path/to/raw_keys.bin <<'PY'
import shutil, struct, sys
from pathlib import Path
source, destination = map(Path, sys.argv[1:])
if source.resolve() == destination.resolve():
    raise ValueError("use a different output path")
with source.open("rb") as src:
    header = src.read(8)
    if len(header) != 8:
        raise ValueError("missing count header")
    count, = struct.unpack("<Q", header)
    if source.stat().st_size != 8 + count * 8:
        raise ValueError("count header does not match file size")
    with destination.open("xb") as dst:
        shutil.copyfileobj(src, dst)
PY
```

To sort raw keys, optionally removing duplicates:

```sh
PARLAY_NUM_THREADS=4 ./build/sort_dataset_lat1_typeNONE_b0_f0 input.bin sorted.bin
PARLAY_NUM_THREADS=4 ./build/sort_dataset_lat1_typeNONE_b0_f0 input.bin unique.bin deduplicate
```

The sorter accepts unsorted and repeated keys, including empty or one-key
inputs. Without `deduplicate`, it preserves repeated keys; remove them before
bulk-loading an index, which requires unique keys.

## Construction stages

```sh
THREADS="1 2 4" B_VALUES="64" F_VALUES="2 4" \
./scripts/run_bench_segmenting.sh 100000
```

This configures the existing build directory without changing its generator,
builds `bench_segmenting_lat0_typeNONE_b0_f0`, and writes a CSV under `results/`.
Use `BUILD_JOBS`, `BUILD_DIR`, and `RESULTS_DIR` to change those defaults.
`NUMA_NODE` and `TIMEOUT_SECONDS` are optional; `PLOT=1` enables plotting and
requires the Python packages in `experiments/requirements.txt`. An input path
uses the raw-key format above.

The plotting tool reports how many CSV rows are usable and how many were
excluded as `ERROR`, `TIMEOUT`, or non-numeric timings. Every generated figure
includes those input-row counts. Failed rows do not contribute to timing means;
the original CSV is preserved. If all rows fail, the counts are printed and the
plot command exits with an error. These are implementation microbenchmarks;
the paper's system figures use the separate supplementary-package workflow.

## Reclamation modes

`FH_INDEX_RECLAMATION` applies only to benchmark targets. Its default is `normal`.

| Mode | Epoch guards | Record retired objects | Automatic reclamation |
|---|---|---|---|
| `normal` | Yes | Yes | Yes |
| `disabled` | No | No | No |
| `guards_only` | Yes | No | No |
| `record_only` | Yes | Yes | No |

Use `normal` for end-to-end FH-index results. The other modes are diagnostic
ablations: replaced allocations are not reclaimed automatically, so their
throughput and memory results must be labeled with the selected mode. CMake
prints the mode and warns for a diagnostic selection. Correctness tests retain
normal reclamation independently of the benchmark setting.

The default allocator is the system allocator. Select jemalloc at configure
time with `-DFH_INDEX_ALLOCATOR=jemalloc` after installing its library; set
`MALLOC_CONF` explicitly for allocator experiments. The scripts do not preload
a machine-specific allocator path. Record the CPU, compiler, worker count,
binding, and allocator when comparing timings.
