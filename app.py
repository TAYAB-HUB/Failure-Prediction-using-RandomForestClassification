"""
Heart Failure Risk Predictor — Streamlit App
Random Forest classifier (class-weighted), threshold tuned for death-event recall.

IMPORTANT: includes a retrospective 'follow-up time' feature — see the
disclaimer in the app. This limits real prospective-prediction validity.
"""

import streamlit as st
import numpy as np
import pandas as pd
import pickle
import json
from datetime import datetime

st.set_page_config(page_title="Heart Failure Risk Predictor", page_icon="❤️", layout="wide")

MODEL_PATH = "heart_failure_model.pkl"
FEATURE_INFO_PATH = "feature_info.json"
DECISION_THRESHOLD = 0.35

BINARY_FEATURES = {"anaemia", "diabetes", "high_blood_pressure", "sex", "smoking"}
FEATURE_LABELS = {
    "age": "Age (years)",
    "anaemia": "Anaemia",
    "creatinine_phosphokinase": "Creatinine Phosphokinase (mcg/L)",
    "diabetes": "Diabetes",
    "ejection_fraction": "Ejection Fraction (%)",
    "high_blood_pressure": "High Blood Pressure",
    "platelets": "Platelets (kiloplatelets/mL)",
    "serum_creatinine": "Serum Creatinine (mg/dL)",
    "serum_sodium": "Serum Sodium (mEq/L)",
    "sex": "Sex",
    "smoking": "Smoking",
    "time": "Follow-up Time (days) ⚠️",
}

# ----------------------------------------------------------------------------
# Cached loaders
# ----------------------------------------------------------------------------
@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_feature_info():
    with open(FEATURE_INFO_PATH, "r") as f:
        return json.load(f)

model = load_model()
feature_info = load_feature_info()
feature_names = feature_info["feature_names"]
feature_stats = feature_info["feature_stats"]

# ----------------------------------------------------------------------------
# Session state
# ----------------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.header("📋 Model Info")
    st.markdown(f"""
    **Model:** Random Forest (100 trees, `class_weight='balanced'`)

    **Trained on:** 299 patients, Heart Failure Clinical Records dataset (UCI)

    **Decision threshold:** {DECISION_THRESHOLD} (tuned to prioritize catching real death-risk cases)

    **Performance (5-fold cross-validated):**
    | Metric | Score |
    |---|---|
    | Death recall | ~70% (±9.9% across folds) |
    | Accuracy | ~85% |

    ⚠️ **Known limitations:**
    - **Small dataset (299 patients)** — cross-validated recall varies substantially fold-to-fold (58%–80%), so treat exact percentages as directional, not precise.
    - **The "Follow-up Time" feature is retrospective, not prospective.** In the original study, it measures how many days a patient was actually followed before death or study end — information a real new patient cannot provide in advance. It is the single strongest predictor in this model. This means the model is better understood as a retrospective outcome-association tool than a true forward-looking clinical risk predictor. Included here for educational/portfolio purposes; a production tool would need to be retrained without it.
    - Not a certified diagnostic or clinical decision-making tool.
    """)

    st.divider()
    st.header("📊 Session History")
    if st.session_state.history:
        for h in reversed(st.session_state.history[-10:]):
            st.write(f"`{h['time']}` — **{h['label']}** ({h['confidence']:.1%})")
    else:
        st.caption("No predictions yet this session.")

    if st.button("Clear history"):
        st.session_state.history = []
        st.rerun()

# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
st.title("❤️ Heart Failure Risk Predictor")
st.caption("Random Forest classifier on clinical follow-up records")

st.warning(
    "⚠️ **Educational project only.** Not a certified diagnostic tool — must not be used for "
    "real medical decisions. See sidebar for an important limitation regarding the "
    "'Follow-up Time' input specifically.",
    icon="⚠️"
)

st.subheader("Patient Clinical Data")

col1, col2, col3 = st.columns(3)
columns = [col1, col2, col3]
inputs = {}

for i, feature in enumerate(feature_names):
    target_col = columns[i % 3]
    stats = feature_stats[feature]
    label = FEATURE_LABELS.get(feature, feature)

    with target_col:
        if feature in BINARY_FEATURES:
            if feature == "sex":
                choice = st.radio(label, ["Female", "Male"], horizontal=True)
                inputs[feature] = 1 if choice == "Male" else 0
            else:
                choice = st.radio(label, ["No", "Yes"], horizontal=True)
                inputs[feature] = 1 if choice == "Yes" else 0
        else:
            min_val = float(stats["min"])
            max_val = float(stats["max"])
            mean_val = float(stats["mean"])
            step = 1.0 if feature in ("age", "creatinine_phosphokinase", "platelets", "time") else 0.1
            inputs[feature] = st.slider(
                label, min_value=min_val, max_value=max_val, value=round(mean_val, 1), step=step
            )

predict_clicked = st.button("Predict Risk", type="primary")

if predict_clicked:
    input_df = pd.DataFrame([[inputs[f] for f in feature_names]], columns=feature_names)
    prob_death = model.predict_proba(input_df)[0][1]
    prob_survive = 1 - prob_death

    label = "HIGH RISK" if prob_death >= DECISION_THRESHOLD else "LOWER RISK"
    confidence = prob_death if label == "HIGH RISK" else prob_survive

    st.divider()
    result_col1, result_col2 = st.columns([1, 1])

    with result_col1:
        if label == "HIGH RISK":
            st.error(f"### Prediction: {label}")
        else:
            st.success(f"### Prediction: {label}")
        st.metric("Confidence", f"{confidence:.1%}")
        st.caption(f"Decision threshold: {DECISION_THRESHOLD} (tuned for recall on death events)")

    with result_col2:
        st.subheader("Probability Breakdown")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5, 2.5))
        classes = ["Survive", "Death Risk"]
        probs = [prob_survive, prob_death]
        colors = ["#2ecc71", "#e74c3c"]
        bars = ax.barh(classes, probs, color=colors)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Probability")
        for bar, p in zip(bars, probs):
            ax.text(p + 0.02, bar.get_y() + bar.get_height() / 2, f"{p:.1%}", va="center")
        ax.axvline(DECISION_THRESHOLD, color="gray", linestyle="--", linewidth=1)
        st.pyplot(fig)

    st.session_state.history.append({
        "time": datetime.now().strftime("%H:%M:%S"),
        "label": label,
        "confidence": confidence
    })
else:
    st.info("Fill in the patient data above and click 'Predict Risk'.")

st.divider()
st.caption(
    "Built with scikit-learn · Random Forest · Streamlit  \n"
    "[GitHub Repo](#) — replace with your repo link"
)
