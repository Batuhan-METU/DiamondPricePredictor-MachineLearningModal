import streamlit as st
import pickle
import pandas as pd

# ---- PAGE SETUP ----
st.set_page_config(
    page_title="Diamond Price Predictor",
    page_icon="💎",
    layout="centered"
)

# ---- CUSTOM STYLES ----
st.markdown("""
    <style>
    body {
        background-color: #f5f7fa;
        font-family: 'Inter', sans-serif;
    }
    .main {
        background-color: #ffffff;
        border-radius: 16px;
        padding: 30px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# ---- LOAD MODEL ----
with open('30-diamond_model_complete.pkl', 'rb') as f:
    saved_data = pickle.load(f)
    model = saved_data['model']
    encoders = saved_data['encoders']
    scaler = saved_data['scaler']

# ---- HEADER ----
st.markdown("## 💎 Diamond Price Predictor")
st.markdown(
    "<p style='color:gray;'>Estimate diamond prices based on key attributes.</p>",
    unsafe_allow_html=True
)
st.divider()

# ---- INPUT FORM ----
col1, col2 = st.columns(2)

with col1:
    carat = st.number_input("Carat", 0.1, 5.0, 1.0)
    cut = st.selectbox("Cut", ["Fair", "Good", "Very Good", "Premium", "Ideal"])
    color = st.selectbox("Color", ["D", "E", "F", "G", "H", "I", "J"])
    clarity = st.selectbox("Clarity", ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"])

with col2:
    depth = st.number_input("Depth", 55.0, 70.0, 61.0)
    table = st.number_input("Table", 50.0, 70.0, 57.0)
    x = st.number_input("x (mm)", 3.0, 10.0, 5.0)
    y = st.number_input("y (mm)", 3.0, 10.0, 5.0)
    z = st.number_input("z (mm)", 2.0, 10.0, 3.0)

st.divider()

# ---- PREDICTION ----
if st.button("💡 Predict Diamond Price", use_container_width=True):
    input_df = pd.DataFrame([{
        "carat": carat,
        "cut": cut,
        "color": color,
        "clarity": clarity,
        "depth": depth,
        "table": table,
        "x": x,
        "y": y,
        "z": z
    }])

    for col in ['cut', 'color', 'clarity']:
        input_df[col] = encoders[col].transform(input_df[col])

    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]

    st.markdown("### 💰 Estimated Diamond Price")
    st.markdown(
        f"<div style='background-color:#f0f8ff;padding:20px;border-radius:12px;text-align:center;font-size:24px;font-weight:600;color:#2b4162;'>${prediction:,.2f}</div>",
        unsafe_allow_html=True
    )
