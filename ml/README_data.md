
# Data Documentation

## 1. Dataset Source

- **Name:** Appliances Energy Prediction
- **Source:** UCI Machine Learning Repository
- **Dataset ID:** 374
- **Original author:** Luis Candanedo
- **Publication year:** 2017
- **DOI:** https://doi.org/10.24432/C5VC8G
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0)
- **Original file:** energydata_complete.csv

## 2. Dataset Description

The dataset contains 19,735 observations collected over approximately 4.5 months, with measurements recorded every 10 minutes.

It includes appliance energy consumption, lighting consumption, indoor temperature and humidity measurements, and outdoor weather data.

## 3. Data Quality Assessment

The initial dataset contains:

- 19,735 observations
- 29 columns
- No missing values
- No duplicated rows
- No duplicated timestamps
- No invalid timestamps
- Regular 10-minute sampling intervals
- Chronologically ordered observations

## 4. Target and Features

- **Target:** Appliances
- **Target unit:** Wh
- **Number of explanatory features:** 25

The date column is retained for chronological ordering but excluded from the model input.

The random variables rv1 and rv2 are excluded from the model inputs.

## 5. Dataset Versioning Strategy

The dataset is divided chronologically into three consecutive periods.

### Dataset v1

- First 60% of observations
- 11,841 observations
- Used for initial model training

### Dataset v2

- First 80% of observations
- 15,788 observations
- Includes the additional historical observations from the 20% extension

### Final Test Set

- Last 20% of observations
- 3,947 observations
- Shared by both v1 and v2 experiments
- Never used for training or hyperparameter tuning

## 6. Reproducibility and Data Versioning

Dataset snapshots are managed using DVC, with DagsHub configured as the remote storage backend.

The original raw dataset and the generated training and evaluation partitions are stored outside Git. Git tracks the corresponding DVC pointer files and metadata.

### DVC Snapshot Identifiers

| Dataset snapshot | DVC file | MD5 |
|---|---|---|
| v1 training | `data/versions/v1/train.csv.dvc` | `b2fc482fe85cd1779a0faa5f7d6de625` |
| v2 training | `data/versions/v2/train.csv.dvc` | `d08e0d2745ace0677a6330d8cc20795a` |
| Final test | `data/versions/test/final_test.csv.dvc` | `082db6d3abab0442e8721d732e1db116` |

### Versioning Strategy

- **v1:** First 60% of the chronological dataset, used for baseline training.
- **v2:** First 80% of the chronological dataset, including the additional historical observations.
- **Final test:** Last 20% of the chronological dataset, shared by both experiments and never used for training or hyperparameter tuning.

The final test partition remains unchanged across both dataset versions to ensure a fair model comparison.

### Git Reproducibility

The Git commit identifiers corresponding to each dataset snapshot will be recorded after committing the DVC pointer files.

The Git commit containing the dataset snapshot pointers and metadata is:
- Commit: `ac07620abb924c56d2ef32183cd6fec055d1f01a`
- Branch: `feature/mlops-pipeline`
