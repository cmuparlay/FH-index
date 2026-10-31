# FH-index internal components

This directory contains implementation shared by the read-only and
read-write indexes. Its subdirectories follow dependency direction:

- `utility/`: low-level macros and arithmetic helpers; does not depend on
  index data structures.
- `memory/`: epoch protection, retirement, and object recycling.
- `representation/`: the shared finger array and finger-group descriptor
  layouts.
- `construction/`: single-level partitioning and multi-level hierarchy
  construction.

Construction may depend on the other three groups. Representation may depend
on memory. Runtime index code should not be pulled into this directory.
Benchmark-only helpers belong in `include/util`, not here.
