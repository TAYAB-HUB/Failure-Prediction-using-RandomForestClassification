# ❤️ Heart Failure Risk Prediction

Predicting patient survival risk from clinical follow-up records using a class-weighted, threshold-tuned Random Forest classifier — deployed as an interactive web app, with an honest look at a real limitation in the underlying dataset.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![scikit--learn](https://img.shields.io/badge/scikit--learn-RandomForest-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-red)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

🔗 **[Live Demo](#)** &nbsp;•&nbsp; 📓 **[Notebook](#)** &nbsp;•&nbsp; 📊 **[Dataset](https://archive.ics.uci.edu/dataset/519/heart+failure+clinical+records)**

---

## 📌 Overview

This project predicts `DEATH_EVENT` (patient survival outcome during follow-up) from 12 clinical features — ejection fraction, serum creatinine, platelet count, comorbidity indicators, and more — using a Random Forest classifier. The goal was to flag high-risk patients from routine clinical measurements, with a deliberate focus on **recall over raw accuracy**: in a clinical risk-screening context, missing a real at-risk patient (a false negative) is a far more costly error than an unnecessary follow-up (a false positive).

**⚠️ Disclaimer:** This is an educational/portfolio project. It is **not** a certified diagnostic or clinical decision-making tool. See [Limitations](#️-limitations) for an important caveat about one of the model's inputs.

---

## 📊 Dataset

| | |
|---|---|
| **Source** | [Heart Failure Clinical Records Dataset (UCI)](https://archive.ics.uci.edu/dataset/519/heart+failure+clinical+records) |
| **Records** | 299 patients |
| **Features** | 12 clinical features: age, anaemia, creatinine phosphokinase, diabetes, ejection fraction, high blood pressure, platelets, serum creatinine, serum sodium, sex, smoking, follow-up time |
| **Target** | `DEATH_EVENT` (1 = death during follow-up, 0 = survived) |
| **Class balance** | 203 survived / 96 death events (~68% / 32% — imbalanced) |
| **Missing values** | None — the dataset is complete; an earlier version of this project applied a forward-fill step defensively, but it had no actual effect since there was nothing to impute |

---

## 🧠 Methodology

### Baseline → improvements

| Step | Change | Why |
|---|---|---|
| 1. Baseline | Plain `RandomForestClassifier`, random 80/20 split | Original approach — 75% accuracy, but only 48% recall on death events |
| 2. Stratified split | Added `stratify=y` to the train/test split | With only 96 death events total, an unlucky random split could skew the test set's class ratio; stratification guarantees both sets preserve the real ~68/32 ratio |
| 3. Class weighting | `class_weight="balanced"` | Directly addresses the imbalance — death recall improved from 48% to 63% at the default 0.5 threshold, with log-loss also improving (0.394 → 0.364) |
| 4. Threshold tuning | Swept thresholds from 0.5 down to 0.2, evaluating recall/precision/accuracy at each | 0.35 gives the best trade-off: recall jumps to 73.7% while *matching* precision (73.7%) and holding accuracy at 83.3% — it dominates lower thresholds, which trade away more precision for the same recall |
| 5. Cross-validation | 5-fold stratified CV on the final model | Confirms the improvement is real, not a lucky test split — but also reveals real fold-to-fold variance (see [Results](#-results)) |

### Final model

**Random Forest (100 trees), `class_weight="balanced"`, decision threshold = 0.35**, retrained on the full 299-patient dataset after evaluation.

---

## 📈 Results

### Single held-out split (60 test patients), before vs after improvements

| Metric | Original baseline | Class-weighted @ threshold 0.35 |
|---|---|---|
| Death recall | 48% | **73.7%** |
| Death precision | 86% | 73.7% |
| Accuracy | 75% | 83.3% |
| Log-loss | 0.394 | 0.364 |

### 5-fold cross-validation (more reliable estimate given the small dataset)

| Metric | Mean | Std dev | Per-fold range |
|---|---|---|---|
| Death recall | 69.7% | ±9.9% | 57.9% – 80.0% |
| Accuracy | 85.3% | ±3.9% | — |

**Honest caveat:** with only 299 patients total, cross-validated recall genuinely varies by over 20 percentage points depending on which patients land in which fold. Treat the headline recall figure as directional (roughly 60-80%), not a precise number — this is a real limitation of the dataset's size, not a modeling flaw to explain away.

---

## ⚠️ Limitations

- **Small dataset (299 patients).** As shown above, this causes real, measurable variance in reported performance across cross-validation folds. Any single-split accuracy/recall number should be read with that variance in mind.
- **The "Follow-up Time" feature is retrospective, not prospective — and it's the model's single strongest predictor.** `time` measures how many days a patient was actually followed before death or the study's end. In the original clinical study this was collected *after the fact* — a real new patient walking into a clinic cannot supply their own future follow-up duration in advance. This means the model, as built, is better understood as a retrospective outcome-association tool rather than a genuine forward-looking risk predictor for a brand-new patient. This is a known, published critique of this exact dataset (see Chicco & Jurman, 2020, the paper that introduced it). Feature importance analysis on the final model confirms `time` dominates: `[fill in your printed importances.head() value here]`.
  - **Decision made:** kept `time` in the deployed model for this educational/portfolio project, with the limitation documented prominently in both this README and directly in the app's input form (labeled with a ⚠️), rather than silently hiding it. A genuinely deployable clinical tool would need to be retrained without this feature.
- Not a certified diagnostic or clinical decision-support tool.

---

## 🖥️ Streamlit App

An interactive app for entering patient clinical data and getting a live risk prediction:

- Input form for all 12 clinical features, with realistic ranges pulled from the training data
- Risk prediction (High Risk / Lower Risk) at the tuned 0.35 threshold, with confidence score
- Probability breakdown chart
- Sidebar with model performance (including the honest cross-validation variance) and the "Follow-up Time" limitation
- Session-based prediction history

![Streamlit App Screenshot](assets/streamlit_app.png)
*Add a screenshot of your running app here.*

### Run locally

```bash
git clone https://github.com/TAYAB-HUB/Heart-Failure-Prediction.git
cd Heart-Failure-Prediction
pip install -r requirements.txt
streamlit run app.py
```

### Deployment

Deployed on **Streamlit Community Cloud**. `runtime.txt` pins Python to 3.11.

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Modeling | scikit-learn (`RandomForestClassifier`, stratified `train_test_split`, `StratifiedKFold`) |
| Evaluation | scikit-learn (`classification_report`, `log_loss`, cross-validation) |
| Data Handling | Pandas, NumPy |
| Visualization | Matplotlib |
| Deployment | Streamlit, Streamlit Community Cloud |

---

## 📁 Project Structure

```
├── app.py                                        # Streamlit web app
├── requirements.txt                              # Python dependencies
├── runtime.txt                                    # Pins Python version for deployment
├── heart_failure_model.pkl                       # Final trained model (class-weighted, threshold 0.35)
├── feature_info.json                             # Feature names + stats, used to build the app's input form
├── Failure Prediction.ipynb                      # Full analysis notebook
├── heart_failure_clinical_records_dataset.csv    # Dataset
├── assets/
│   └── streamlit_app.png                         # Screenshot used in this README
└── README.md
```

---

## 🚀 Future Improvements

- Retrain a second version **without** the `time` feature, to compare how much performance depends on that retrospective signal versus genuinely prospective clinical measurements
- Try SMOTE oversampling as an alternative to `class_weight="balanced"`, and compare
- Feature scaling + comparison against Logistic Regression / Gradient Boosting baselines
- Hyperparameter tuning (`GridSearchCV`) on `n_estimators`, `max_depth`, `min_samples_split`
- Explore whether a larger, multi-institution dataset reduces the cross-validation variance seen here

---

## 📄 License

This project is licensed under the MIT License.

---

## 🙋 Author

**Syed Mohammed Tayab** — CSE student, Presidency University, Bengaluru
[GitHub](https://github.com/TAYAB-HUB) &nbsp;•&nbsp; [LinkedIn](https://linkedin.com/in/syed-tayab01)
