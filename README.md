# Road Accident Severity Prediction

A small ML project that predicts how severe a road accident is likely to be (Fatal / Serious / Slight) using the UK Department for Transport's road collision data. It uses XGBoost for the actual classification and SHAP to figure out which features matter, then retrains a slimmer model using just those.

I built this as my minor project, mainly because I wanted to work with a real, messy, imbalanced dataset instead of another clean Kaggle CSV.

## What's in the notebook

- Loads the DfT road casualty statistics dataset (2024 collisions, ~100k rows)
- Drops columns that would leak the outcome (things like number of casualties) or are just IDs/location codes
- Engineers a handful of extra features - rush hour, night time, weekend, wet road, bad weather
- Encodes categorical columns, fills in missing values
- Splits into train / validation / test
- Trains a baseline XGBoost model
- Runs SHAP on it to rank feature importance
- Retrains using only the top 15 SHAP features and compares the two models
- Includes a small `predict_severity_selected_features()` function you can pass a dictionary of raw values into and get a prediction back

## Dataset

Using the 2024 collision file from DfT's open data:
https://data.dft.gov.uk/road-accidents-safety-data/dft-road-casualty-statistics-collision-2024.csv

It's not in this repo (it's ~100k rows), so you'll need to grab it yourself. The notebook was written for Colab and mounts Google Drive to read it - update `CSV_PATH` if you're running it somewhere else.

If it can't find the file, it falls back to generating a synthetic dataset with the same columns so the pipeline still runs end to end. Handy if you just want to see the code work without downloading anything.

## Tools

Python, pandas, numpy, XGBoost, SHAP, scikit-learn, matplotlib, seaborn.

## Results

| Model | Accuracy | Balanced Accuracy | Macro F1 | Weighted F1 |
|---|---|---|---|---|
| Baseline XGBoost (27 features) | 0.7515 | 0.3359 | 0.2933 | 0.6500 |
| XGBoost + SHAP feature selection (15 features) | 0.7512 | 0.3353 | 0.2922 | 0.6491 |

Cutting the feature set almost in half barely moved the numbers, which I take as a decent sign - most of those other columns weren't doing much anyway.

SHAP ranked whether a police officer attended the scene, number of vehicles, speed limit, and urban vs. rural area as the most important features, by a fair margin over everything else.

**Being honest about the results:** accuracy looks okay at ~75%, but that's mostly the class imbalance talking. Around 75% of collisions in the data are "Slight," so the model leans hard into predicting that. It gets 0% recall on "Fatal" and only about 1% recall on "Serious" - it's basically not learning to separate the severe cases from the rest. Wanted to flag that clearly instead of just posting the accuracy number and moving on.

## What I'd do differently with more time

- Actually deal with the class imbalance - class weights, SMOTE, or splitting it into a two-stage problem (severe vs. not severe, then Fatal vs. Serious)
- Use a cost-sensitive loss so missing a Fatal case is penalized harder than missing a Slight one
- Proper cross-validation instead of one train/val/test split
- Real hyperparameter search instead of hand-picked values

## Running it

The notebook: **[paste your Colab/notebook link here]**

Or locally:

```bash
pip install pandas numpy xgboost shap scikit-learn matplotlib seaborn
jupyter notebook Minor_project.ipynb
```

Just swap out the Google Drive mount for a normal file path if you're not using Colab, and point `CSV_PATH` at wherever you saved the dataset.

---

Data belongs to the UK Department for Transport - more on their [road safety statistics page](https://www.gov.uk/government/statistical-data-sets/road-safety-statistics-data-tables).
