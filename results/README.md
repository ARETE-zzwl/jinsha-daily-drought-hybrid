# Results Layout

`example_run/` contains compact outputs from the manuscript's main training run.
These files are useful for checking reported metrics without rerunning the full
training workflow.

New runs are written to `runs/` by default. The `runs/` directory is ignored by
Git because it can contain checkpoints and large prediction tables.
