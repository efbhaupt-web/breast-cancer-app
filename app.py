import streamlit as st
import joblib
import pandas as pd

# -----------------------
# Configuración de página
# -----------------------
st.set_page_config(
    page_title="Predicción de cáncer",
    page_icon="🧬",
    layout="centered"
)

# -----------------------
# Cargar modelo
# -----------------------
model = joblib.load("modelo.pkl")
cols = joblib.load("columnas.pkl")

# -----------------------
# HEADER
# -----------------------
st.title("🧬 Predicción de cáncer de mama")
st.markdown("### Modelo de Machine Learning aplicado a datos clínicos")

st.markdown("---")

# -----------------------
# INFO
# -----------------------
st.info(
    "Esta herramienta utiliza un modelo de regresión logística entrenado "
    "para clasificar tumores como benignos o malignos."
)

# -----------------------
# VARIABLES CLAVE
# -----------------------
st.subheader("📊 Parámetros del paciente")

col1, col2 = st.columns(2)

with col1:
    worst_radius = st.number_input("Worst radius", value=15.0)
    worst_perimeter = st.number_input("Worst perimeter", value=100.0)
    mean_perimeter = st.number_input("Mean perimeter", value=80.0)

with col2:
    worst_concave_points = st.number_input("Worst concave points", value=0.1)
    mean_concave_points = st.number_input("Mean concave points", value=0.05)

# -----------------------
# BOTÓN
# -----------------------
st.markdown("---")

if st.button("🔍 Analizar tumor"):

    # Armar input completo
    inputs_dict = {
        "worst radius": worst_radius,
        "worst perimeter": worst_perimeter,
        "mean perimeter": mean_perimeter,
        "worst concave points": worst_concave_points,
        "mean concave points": mean_concave_points
    }

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

    # RESULTADO
    if pred[0] == 0:
        st.error("⚠️ Resultado: TUMOR MALIGNO")
    else:
        st.success("✅ Resultado: TUMOR BENIGNO")

    # PROBABILIDAD
    st.subheader("📈 Probabilidad")

    st.write(f"Benigno: {prob[1]*100:.2f}%")
    st.write(f"Maligno: {prob[0]*100:.2f}%")

    st.progress(float(prob[1]))

    # INTERPRETACIÓN
    st.markdown("---")
    st.subheader("🧠 Interpretación")

    st.write(
        "El modelo evalúa características morfológicas del tejido. "
        "Valores elevados en parámetros como concavidad o perímetro "
        "pueden asociarse con mayor riesgo de malignidad."
    )

# -----------------------
# FOOTER
# -----------------------
st.markdown("---")
st.caption("Proyecto de Machine Learning aplicado a biomedicina")