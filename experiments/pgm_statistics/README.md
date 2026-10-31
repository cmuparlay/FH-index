# PGM Statistics

This experiment measures how PGM-index's last-mile-search bound (`epsilon`)
affects partition-finding time, last-mile-search time, partition count, height,
and memory consumption on real datasets.

The maintained implementation does not generate synthetic data. The
`uniformrandom_200M_uint64.bin` workload is treated like every other input
file.

## Dataset and Query Semantics

Input files use the SOSD binary format:

- the first 8 bytes are a little-endian `uint64_t` key count;
- the remaining bytes are that many sorted little-endian `uint64_t` keys.

`--analysis START END` selects a zero-based half-open rank range `[START, END)`
to index. Bounds accept `K`, `M`, and `B` suffixes.

Queries are sampled uniformly by rank, with replacement, from all indexed keys.
Consequently every measured query is a successful lookup. The same query vector
is reused for every epsilon within a dataset.

Upstream PGM-index reserves `UINT64_MAX` as an internal sentinel and rejects an
input ending in that value. Following the upstream test convention, the runner
excludes trailing `UINT64_MAX` values from PGM construction and query sampling,
and records both `pgm_indexed_keys` and `excluded_sentinel_keys` in every CSV and
in the run metadata. Among the default datasets, only `fb` is affected: one of
its 200M keys is excluded.

## Timing Protocol

Each `--runs` unit is a counterbalanced pair:

1. measure partition finding, then total lookup;
2. measure total lookup, then partition finding.

Both paths are first warmed once in both orders using the full query vector:
partition finding, total lookup, total lookup, then partition finding. The two
partition-finding measurements and the two total-lookup measurements are averaged
within each measured group. Reported means and sample standard deviations are
then computed over the groups.

`last_mile_time_ns` is the paired estimate
`total_time_ns - find_partition_time_ns`; it is not timed with a per-query clock.

## Full experiment

```bash
numactl --physcpubind=0 --membind=0 \
  .venv/bin/python experiments/pgm_statistics/pgm_statistics.py \
  --analysis 0 200M \
  --data-dir /path/to/data \
  --out-dir results/pgm_full \
  --num-queries 100M \
  --runs 3 \
  --seed 42 \
  --warmup-queries 100M \
  --epsilons 2 4 8 16 32 64 128 256 512 \
  --save-pdf \
  --contact-sheet
```

The default dataset list is the same eleven 200M-key datasets used by
`noise_accumulation`. Clang++ is the default compiler; the paper used Clang++ 18.1.3. The command pins the serial
benchmark to CPU 0 and allocates its memory from NUMA node 0.

Use `--mode benchmark` to generate CSV files without plotting or `--mode plot`
to regenerate figures from existing summaries. `--resume` skips datasets whose
raw and summary CSV files are already complete.

## Outputs

The output directory contains:

- `raw/<dataset>.csv`: every forward/reverse measurement group;
- `summary/<dataset>.csv`: per-epsilon means, standard deviations, and index
  statistics;
- `summary.csv`: all per-dataset summaries;
- `figures/<dataset>.png` and optional PDFs;
- `run_metadata.json`: inputs, compiler, PGM commit, protocol, and source hashes;
- an optional contact sheet.

`experiments/pgm_statistics/build/` and `results/` are completely ignored by
Git.
