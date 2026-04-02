**Files definitions**

This folder contains the earlier extraction and preprocessing scripts from the
previous workflow. The Python files are stored directly in this folder; there
is no `scripts/` subfolder in this archive.

Generated CSV data was intended to live outside this archived folder, for
example under the main workspace `Data/` directory.

**Running scripts**

0. Use Python 3.12+ and install dependencies with:
```bash
pip install -r requirements.txt
```

1. Run `extract-alerts.py` to build an alerts CSV such as `alerts_data.csv`.

2. Run `extract-bugs-api.py` after extracting alerts to build a bugs CSV such
as `bugs_data.csv`.

3. Run `extract-timeseries.py` to download timeseries CSV files.

4. Run `transform-data.py` to cross-reference timeseries CSVs with the alerts
CSV.

5. Optional preprocessing helpers:
```text
smoothe.py
minmaxscale.py
aggregate.py
handpick_specific_files.py
jsonfy-timeseries.py
```

**Note**

For data consistency, extract alerts and timeseries close together in time.
Otherwise older alerts may not match downloaded timeseries, and newer alerts
may be missing from an older alerts CSV.
