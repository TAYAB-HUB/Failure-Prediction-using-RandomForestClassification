# Heart Failure Prediction — Random Forest Classifier

Predicting patient survival risk from clinical records using a Random Forest classifier, built on the Heart Failure Clinical Records dataset. The goal is to flag high-risk patients from routine clinical measurements to support earlier, more proactive care decisions.

## Overview

This project applies a supervised classification pipeline to predict `DEATH_EVENT` (patient survival outcome) from 12 clinical features collected during follow-up, including ejection fraction, serum creatinine, platelet count, and comorbidity indicators like anaemia, diabetes, and high blood pressure.

## Dataset

- **Source:** [Heart Failure Clinical Records Dataset (UCI / Kaggle)](https://archive.ics.uci.edu/dataset/519/heart+failure+clinical+records)
- **Records:** 299 patients
- **Features:** 12 clinical features (age, anaemia, creatinine phosphokinase, diabetes, ejection fraction, high blood pressure, platelets, serum creatinine, serum sodium, sex, smoking, follow-up time)
- **Target:** `DEATH_EVENT` (1 = death during follow-up period, 0 = survived)

## Tech Stack

- Python
- Pandas
- scikit-learn (`RandomForestClassifier`, `train_test_split`, `classification_report`, `confusion_matrix`)

## Pipeline

1. **Data loading & inspection** — loaded the CSV and checked for missing values
2. **Missing value handling** — forward-fill imputation (`ffill`)
3. **Train/test split** — 80/20 split, `random_state=42`
4. **Model training** — `RandomForestClassifier` with 100 estimators
5. **Evaluation** — confusion matrix and classification report on the held-out test set

## Results

Evaluated on a 60-sample held-out test set:

| Metric | Class 0 (Survived) | Class 1 (Death) |
|---|---|---|
| Precision | 0.72 | 0.86 |
| Recall | 0.94 | 0.48 |
| F1-score | 0.81 | 0.62 |

**Overall accuracy: 75%**

```
Confusion Matrix:
[[33  2]
 [13 12]]
```

### Interpretation

- The model is strong at correctly identifying patients who survive (94% recall on class 0) and has high precision when it does predict death (86%) — false alarms are rare.
- Recall on the death class is the main weakness (48%) — the model misses roughly half of actual death events, likely due to class imbalance (203 survived vs. 96 death events in the full dataset) and the relatively small sample size.
- For a clinical risk-screening use case, this trade-off matters: high precision but low recall means the model is conservative and under-flags at-risk patients rather than over-flagging healthy ones.

## Future Improvements

- Address class imbalance with `class_weight='balanced'` or SMOTE oversampling
- Feature scaling and comparison against Logistic Regression / Gradient Boosting baselines
- Hyperparameter tuning (`GridSearchCV`) on `n_estimators`, `max_depth`, `min_samples_split`
- Feature importance analysis to identify the strongest clinical predictors
- Cross-validation instead of a single train/test split, given the small dataset size

## Project Structure

```
├── Failure_Prediction.ipynb                    # Main notebook: preprocessing, training, evaluation
├── heart_failure_clinical_records_dataset.csv  # Dataset
└── README.md
```

## Usage

```bash
pip install pandas scikit-learn
jupyter notebook Failure_Prediction.ipynb
```

## Author

**Syed Mohammed Tayab**
[GitHub](https://github.com/TAYAB-HUB) | [LinkedIn](https://linkedin.com/in/syed-tayab01)
