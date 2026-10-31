# Paper datasets

Download the paper inputs separately from
[FH-index datasets on Kaggle](https://www.kaggle.com/datasets/hobopia/fh-index-datasets).
The post-revision artifact uses version 10. Dataset redistribution follows the
source dataset's terms; the FH-index code license does not relicense the data.

The [manifest](manifest.json) records file names, key counts, sizes and SHA-256
hashes copied from the paper's post-revision supplementary artifact. These hashes
identify the inputs; this checkout does not contain the large binary datasets.

The 200M-key real datasets are Books, COVID, FB, Genome, History, Libio, OSM,
Planet, Stack, and WISE. The synthetic inputs are Uniform, Lognormal, Mixture,
and Sawtooth. The larger suite contains 800M-key Uniform, COVID, and OSM.
The three imported analysis drivers default to Uniform and the ten real datasets;
use their `--datasets` option to select other files.

## Format

Paper analysis inputs use little-endian SOSD format:

```text
uint64 key_count
uint64 keys[key_count]
```

Keys are sorted. A 200M-key file is 1,600,000,008 bytes. PGM reserves
`UINT64_MAX`; the PGM analysis records omitted sentinel keys, and partition-count
comparison removes the sentinel and duplicates from both methods.

The implementation benchmarks under `benchmark/` use raw `uint64` keys without
the count header. See [conversion instructions](../docs/benchmarks.md#dataset-format).
