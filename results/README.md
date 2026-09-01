# Results Layout

`example_run/` contains compact outputs from the manuscript's main training run.
These files are useful for checking reported metrics without rerunning the full
training workflow.

New runs are written to `runs/` by default. The `runs/` directory is ignored by
Git because it can contain checkpoints and large prediction tables.

The Zenodo archive also contains `results/archived/jinsha_no_station_embedding/`,
which supplies the two Jinsha-trained checkpoints and stacking metadata required
by the zero-shot Upper Yellow River evaluation. Full Upper Yellow River prediction
tables, checkpoints, and compact metrics are archived in a separate Zenodo ZIP.
