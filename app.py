import streamlit as st
import joblib
import pandas as pd

# -----------------------
# CONFIG
# -----------------------
st.set_page_config(
    page_title="Breast Cancer Prediction",
    page_icon="🧬",
    layout="centered"
)

# -----------------------
# LOAD MODEL
# -----------------------
try:
    model = joblib.load("modelo.pkl")
    cols = joblib.load("columnas.pkl")
except Exception as e:
    st.error(f"Error loading model files: {e}")
    st.stop()

# -----------------------
# HEADER
# -----------------------
st.title("🧬 Breast Cancer Prediction")

st.markdown("""
### 🧠 What does this tool do?

This model predicts whether a tumor is **benign or malignant**  
based on morphological cellular features.

⚠️ **Important:** This is an educational model based on the Wisconsin dataset.  
It does NOT replace professional medical diagnosis.
""")

st.markdown("---")

# -----------------------
# INPUTS
# -----------------------
st.subheader("📊 Patient Parameters")

col1, col2 = st.columns(2)

with col1:
    worst_radius = st.number_input("Worst Radius", value=15.0)
    worst_perimeter = st.number_input("Worst Perimeter", value=100.0)
    mean_perimeter = st.number_input("Mean Perimeter", value=80.0)

with col2:
    worst_concave_points = st.number_input("Worst Concave Points", value=0.1)
    mean_concave_points = st.number_input("Mean Concave Points", value=0.05)

st.markdown("---")

# -----------------------
# PREDICTION
# -----------------------
if st.button("🔍 Analyze Tumor"):

    inputs_dict = {
        "worst radius": worst_radius,
        "worst perimeter": worst_perimeter,
        "mean perimeter": mean_perimeter,
        "worst concave points": worst_concave_points,
        "mean concave points": mean_concave_points
    }

    try:
        full_input = []

        for col in cols:
            if col in inputs_dict:
                full_input.append(inputs_dict[col])
            else:
                full_input.append(0)

        nuevo_df = pd.DataFrame([full_input], columns=cols)

        pred = model.predict(nuevo_df)
        prob = model.predict_proba(nuevo_df)[0]

        st.markdown("---")

        # RESULT
        if pred[0] == 0:
            st.error("⚠️ Result: High probability of MALIGNANT tumor")
        else:
            st.success("✅ Result: High probability of BENIGN tumor")

        # PROBABILITIES
        st.subheader("📈 Model Confidence")

        benign_prob = prob[1] * 100
        malignant_prob = prob[0] * 100

        st.write(f"Benign: {benign_prob:.2f}%")
        st.write(f"Malignant: {malignant_prob:.2f}%")

        st.progress(float(prob[1]))

        # INTERPRETATION
        st.markdown("---")
        st.subheader("🧠 Interpretation")

        st.write("""
        Higher values in parameters such as **concavity** and **perimeter**  
        are associated with greater structural irregularity,  
        which may correlate with malignancy in this dataset.
        """)

    except Exception as e:
        st.error(f"Prediction error: {e}")

# -----------------------
# FOOTER
# -----------------------
st.markdown("---")
st.caption("Machine Learning Project applied to biomedical data | Educational demo")
