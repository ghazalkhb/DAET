# Data collection and preprocessing scripts

This folder contains the code to extract, transform, and analyze performance-related data under `scripts`. The transformations are statistical tranformations such as minmax scaling. The generated data resides in the `datasets` folder.
The code files directly under `scripts` folder have definitions that is obtained by running `python <python_file_path> --help`

## Running the scripts

0. From the repository root, create an isolated environment and install the
   pinned dependencies shared with the evaluation:
```
python -m pip install -r Data/scripts/requirements.txt
```

1. In order to run the script to extract the alerts, run `extract-alerts.py`
It will generate the aelrt data CSV (for example `alerts_data.csv`) which will have the performance alerts data from the time of running the script all the waty back to one year before that.

2. Once you have the alerts CSv, you can extract their associated bugs. In order to run the script to extract the bugs, run `extract-bugs-api.py`
It will generate the bugs CSV (for example `bugs_data.csv`) which will have all the bugs associated with the alerts extracted inthe earlier alerts CSV.

3. To extract the time-series data, run `extract-timeseries.py`.
Note that for the case of autoland, timeseries files were divided across multiple folders because github does not support pushing a folder with more than 1000 files in them. The splitting process took place manually.

> **Note**
> In case the alerts CSV and the timeseries data extraction are not done back-to-back, there might be situation where some of the oldest alerts will not cross-reference with any of the timeseries. There could be also a situation where new alerts got triggered that would cross-rzference with a specific timeseries but the problem is that the new alerts doesn't exist in the alerts CSV (given that they were extracted earlier than the alert creation). So, for data consistency purposes, please extract the alerts CSV and the timeseries data back-to-back.

4. Proceed with cross-referencing the timeseries CSVs with the alerts CSV.n order to run the script to extract the alerts, run `transform-data.py`. Note that the scripts contains the folders mapping to projects because autoland has 4 associated folders as mentionned earlier.

5. Optional: you can run data transformations on the timeseries (smoothing using `smoothe.py`, minmax scaling using `minmaxscale.py`, aggregation using `aggregate.py`, or a combination of some of them). This will output CSV files same as the previously extracted timeseries ones, with a change only occurring in the measurements. In case you want to build an aggregation method of the measurements, insipre fro mhe structure of the mentioned files.

6. The timeseries data could be extensive and only a subset of it is needed to perform initial analysis for example. So, using `handpick_specific_files.py`, you could isolate only specific timeseries.
