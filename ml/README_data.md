# Data Documentation

This document records the source, license, versioning, split design, and
reproducibility metadata for the datasets used by `enercast-mlops`.

## 1. Dataset source and license

- **Name:** Appliances Energy Prediction
- **Source:** UCI Machine Learning Repository
- **Dataset ID:** 374
- **DOI:** https://doi.org/10.24432/C5VC8G
- **Original author:** Luis Candanedo
- **Publication year:** 2017
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0)
- **Original file:** `energydata_complete.csv`
- **Usage terms:** attribution to the original author and UCI is required.
  The derived snapshots below inherit the CC BY 4.0 terms.

The exact source snapshot is identified by the raw DVC pointer and MD5 hash in
section 6. The raw CSV itself is never committed to Git.

## 2. Dataset description

The dataset contains 19,735 observations collected in a single household over
approximately 4.5 months (2016-01-11 to 2016-05-27), with one measurement every
10 minutes.

It contains 29 columns:

- `date`: observation timestamp;
- `Appliances`: total appliance energy consumption, the prediction target, in
  Wh;
- `lights`: light fixture energy consumption, in Wh;
- `T1`-`T9` and `RH_1`-`RH_9`: indoor temperature (degrees Celsius) and relative
  humidity (percent) for nine house zones;
- `T_out`, `Press_mm_hg`, `RH_out`, `Windspeed`, `Visibility`, `Tdewpoint`:
  outdoor weather station measurements. The API and the model use the names
  `To` and `Pressure` for outdoor temperature and atmospheric pressure;
- `rv1`, `rv2`: random noise variables, excluded from the model inputs.

The raw dataset has 25 explanatory features. `ml/preprocess.py` adds `hour` and
`day_of_week` derived from `date`, so the model receives 27 inputs. The `date`
column is kept for chronological ordering but is not a model input.

## 3. Data quality assessment

The raw dataset has:

- 19,735 observations and 29 columns;
- no missing values;
- no duplicated rows;
- no duplicated timestamps;
- no invalid timestamps;
- regular 10-minute sampling intervals;
- chronologically ordered observations.

The pipeline enforces the contract again before any training or promotion
(`ml/preprocess.py`, `ml/splits.py`, `ml/pipeline.py`):

- `date` must be present, parseable as `%Y-%m-%d %H:%M:%S`, unique, and sorted
  in ascending order;
- the target `Appliances` must be numeric, finite, and non-negative;
- every model feature must be numeric and finite;
- the training and validation split must be chronological.

A violation stops the run before any model is registered or promoted.

## 4. Split design

All splits are chronological. The data is never shuffled, so the model never
learns from future observations.

The full dataset is divided into three consecutive periods:

| Snapshot | Period | Rows | Role |
| --- | --- | ---: | --- |
| `v1` | first 60% | 11,841 | initial baseline training pool |
| `v2` | first 80% | 15,788 | continuous-training pool (v1 is a prefix of v2) |
| final test | last 20% | 3,947 | independent reporting set, shared by v1 and v2 |

Within each training pool, the last 15% of the rows (rounded down) form the
validation split (`VALIDATION_FRACTION = 0.15` in `ml/splits.py`). The validation split is later
than the training split, so the champion and the candidate are always compared
on observations that follow their fitting period.

| Training pool | Train rows | Validation rows | Train period | Validation period |
| --- | ---: | ---: | --- | --- |
| `v1` | 10,065 | 1,776 | 2016-01-11 17:00 to 2016-03-21 14:20 | 2016-03-21 14:30 to 2016-04-02 22:20 |
| `v2` | 13,420 | 2,368 | 2016-01-11 17:00 to 2016-04-13 21:30 | 2016-04-13 21:40 to 2016-04-30 08:10 |

The final test covers 2016-04-30 08:20 to 2016-05-27 18:00. It is never read by
the training, comparison, or promotion path. It is only used by
`python -m ml.pipeline --final-evaluation` for reporting after model selection.

The v2 validation period starts after the end of the v1 training pool, so the
v1 champion has not seen the observations used to compare it with the v2
candidate. This is the key property that makes the continuous-training
comparison fair.

## 5. Dataset versions and changes

| Version | Change |
| --- | --- |
| raw | Original UCI snapshot, kept as the immutable source. |
| `v1` | First usable chronological snapshot: first 60% of the observations. Used to train and promote the initial baseline champion. |
| `v2` | Extends `v1` with the next 20% of observations (chronologically after `v1`). Same columns, target, cleaning rules, and feature engineering; no relabeling and no modification of the `v1` rows. This simulates a new data delivery that can trigger continuous training. |
| final test | Last 20% of the observations, fixed and shared by both versions. Never used for training or tuning. |

`v2` is a superset of `v1`, not a replacement: the first 60% of the rows are
identical. The only change is the additional later observations. The final test
remains unchanged across both versions, so the comparison between the baseline
and the candidate is made on the same held-out period.

## 6. DVC snapshots and reproducibility

DVC pointers are committed to Git; the CSV files are listed in the local
`.gitignore` files and are never committed. The DVC remote is DagsHub:

```text
https://dagshub.com/MezziNour/enercast-mlops.dvc
```

| Snapshot | DVC pointer | MD5 | Git commit |
| --- | --- | --- | --- |
| raw | `data/raw/energydata_complete.csv.dvc` | `69ef922b5fcafcd49097cfc09e07167e` | `55a871a955d091cbb031e07c272fc812ea4f62ad` |
| `v1` | `data/versions/v1/train.csv.dvc` | `b2fc482fe85cd1779a0faa5f7d6de625` | `55a871a955d091cbb031e07c272fc812ea4f62ad` |
| `v2` | `data/versions/v2/train.csv.dvc` | `d08e0d2745ace0677a6330d8cc20795a` | `55a871a955d091cbb031e07c272fc812ea4f62ad` |
| final test | `data/versions/test/final_test.csv.dvc` | `082db6d3abab0442e8721d732e1db116` | `55a871a955d091cbb031e07c272fc812ea4f62ad` |

All four pointers entered `dev` in squash merge `55a871a` (#6).
The pipeline reads the MD5 from each `.dvc` pointer and verifies the
file bytes before use (`ml/pipeline.py`), so a modified or partially pulled file
aborts the run.

### Local reproduction

```bash
uv sync --frozen --group experiment
# DVC remote credentials: see CONTRIBUTING.md, "DVC Remote"
uv run dvc pull data/versions/v1/train.csv.dvc
uv run dvc pull data/versions/v2/train.csv.dvc
uv run dvc pull data/versions/test/final_test.csv.dvc
```

The `ct-retrain.yml` workflow configures the same credentials from the
`MLFLOW_TRACKING_USERNAME` variable and the `MLFLOW_TRACKING_PASSWORD`
secret, then pulls only the pointer selected by the workflow input.

## 7. Data contract

The pipeline requires:

- `date` in `%Y-%m-%d %H:%M:%S`, unique and strictly increasing;
- `Appliances` numeric, finite, and non-negative;
- `lights`, `T1`-`T9`, `RH_1`-`RH_9`, `To`, `Pressure`, `RH_out`, `Windspeed`,
  `Visibility`, and `Tdewpoint` present and numeric;
- no missing or non-finite feature values;
- a chronological train/validation split with at least one row on each side.

The API contract in `docs/API.md` uses the same feature names and physical
ranges. The training pipeline accepts the original UCI names `T_out` and
`Press_mm_hg` and normalizes them to `To` and `Pressure`.

## 8. Usage in the pipeline

- `ml.pipeline.load_data(version)` reads only `data/versions/<version>/train.csv`
  and splits it chronologically. It never reads the final test.
- `ml.pipeline.load_final_test_data()` is reserved for
  `python -m ml.pipeline --final-evaluation`.
- `ct-retrain.yml` pulls only the selected training pointer; it does not pull
  the final test.
- The final test is used for reporting only. It must never influence model
  selection, hyperparameter tuning, or promotion.

## 9. Known limitations

- The target is noisy and highly skewed (median 60 Wh, mean about 98 Wh). The
  primary metric is MAE in Wh; see `README_experiments.md` for the rationale
  and the quality gates.
- The data covers one household over approximately 4.5 months. Generalization
  to other buildings or seasons is limited.
- `v1` and `v2` overlap by design. The final test is shared by both versions
  and must remain untouched by model selection.
- `v2` simulates a new delivery by extending the chronological window; it does
  not add new labels, new sensors, or corrected labels.
