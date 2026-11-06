# Bitstream statistics

Run statistics on one experiment at a time:

```bash
python tools/stream_stats.py /path/to/tffc/streams --branches 1
python tools/stream_stats.py /path/to/affc/streams --branches 3
```

The tool groups streams by frame ID and requires one geometry and one attribute file for each branch. T-FFC uses one branch; A-FFC uses three. Geometry streams use `_C.bin` and attribute streams use `_F.bin`. Intermediate NPY/PLY files are excluded from the calculation.

Reports contain the number of complete frames and mean geometry, attribute and total bytes per frame, also expressed as decimal kB and KiB. Codec headers are included in file totals. Network packet framing and externally supplied model configuration/calibration are outside these file totals.

The encoder and decoder receive the same fixed spatial grid from configuration. For older streams with a three-byte application spatial header, pass `--legacy-shape-header`.

Feature point/channel reduction, tensor memory size, saved-array file size and encoded bitstream size measure different quantities. For a separately measured baseline, `--baseline-bytes-per-frame` computes its ratio to the mean encoded bytes per frame; use the same sample set and accounting convention for both measurements.
