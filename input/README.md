# Milestone 3 input definitions

This directory contains small, versioned definitions used to reproduce
Milestone 3. It does **not** contain the crawled corpus, M2 outputs, labels,
models, or other generated data.

- `member_ranges.csv` records the five ranges assigned by the lecturer.
- `member-paper-ids/` contains the frozen, authoritative ID list for each
  lecturer-assigned range.
- `input_manifest.json` is path-portable: its paths are relative to a runtime
  data root supplied through `--data-root` or `M3_DATA_ROOT`.

The actual frozen-data manifest with row counts and checksums belongs outside
the repository at `<data-root>/m3/frozen-inputs/input_manifest.json`.
