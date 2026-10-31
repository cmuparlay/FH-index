# Partition-count reference and release recheck

## Current paper figure data

`paper_counts.csv` is the corrected dataset used by the manuscript figure as of
2026-11-01. It uses the full-size recheck below: COVID has 42,695 PGM partitions
and WISE has 27,553. All FH counts are unchanged. The indexed-key counts are
also corrected after explicit SOSD-header and sentinel handling.

Render the current paper figure from the repository root:

```sh
.venv/bin/python experiments/partition_count/plot_paper.py \
  --input experiments/partition_count/reference/paper_counts.csv \
  --output results/paper/partition_count_comparison.pdf
```

## Historical paper values

`paper_counts_20260901_legacy.csv` contains the historical values from the frozen
`20260901_partition_count_final` supplementary measurement. Only absolute input
paths were replaced with file names. Parameters: PGM epsilon 64; FH B=128, F=4;
Uniform plus ten real 200M-key datasets.

The archived run used FH commit `7e5202dd6619c043c40745404b772d6a85e26d23`,
PGM commit `c6fcf3d34e55eb0061b01e2f49dfcbdb711f1407`, GCC 13.3.0, and 192 workers
on NUMA node 0. These rows remain unchanged as historical observations.

## Full-size recheck on 2026-11-01

The current partition-count implementation was run on all eleven full-size
inputs on genoa3, with the same PGM version, parameters, compiler version, and
NUMA/thread configuration. Every input SHA-256 matched `datasets/manifest.json`.
The current implementation reads the SOSD count header explicitly.

- FH partition counts match on all 11 inputs.
- PGM partition counts match on 9 inputs.
- COVID: 42,696 in the archived run, 42,695 with the corrected reader.
- WISE: 27,554 in the archived run, 27,553 with the corrected reader.

A controlled check kept the current algorithm and compiler settings and changed
only the dataset reader back to the old raw-word reader. It reproduced 42,696
for COVID and 27,554 for WISE, with one FH partition in both cases. The two
one-partition differences therefore come from the input-reader behavior.

The old reader treated the SOSD count word as a key and, at a 200M-word load
limit, omitted the final real key. The corrected reader also changes recorded
indexed-key counts: FB becomes 199,999,999 after removing its reserved sentinel;
Libio and Stack become 200,000,000. Their partition counts are unchanged.

Files:

- `recheck_20261101.csv`: old/new counts and integer deltas for every input.
- `legacy_reader_ab_20261101.csv`: controlled old-reader results for COVID and WISE.
- `recheck_environment_20261101.json`: input hashes, compiler/placement, dependencies,
  and source-file hashes identifying the tested implementation.

Use the corrected-reader results for new experiments. The archived paper values
are preserved separately so changes in input handling are not hidden.
