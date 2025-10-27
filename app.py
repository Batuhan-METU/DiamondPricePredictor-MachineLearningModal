import streamlit as st
import pickle
import pandas as pd

# ---- PAGE CONFIG ----
st.set_page_config(
    page_title="Diamond Price Predictor 💎",
    page_icon="💎",
    layout="centered"
)

# ---- MODERN CSS STYLING ----
st.markdown("""
    <style>
    /* Background Gradient */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        background-attachment: fixed;
        color: #fff;
    }

    /* Glass card effect */
    .main {
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(15px);
        -webkit-backdrop-filter: blur(15px);
        border-radius: 20px;
        padding: 40px;
        margin-top: 40px;
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
    }

    /* Title styling */
    h2, h3 {
        color: #ffffff;
        text-align: center;
        font-weight: 600;
    }

    /* Input labels */
    label, .stSelectbox label {
        font-weight: 600;
        color: #f1f1f1;
    }

    /* Buttons */
    div.stButton > button {
        background: linear-gradient(90deg, #36d1dc 0%, #5b86e5 100%);
        color: white;
        border: none;
        padding: 0.7rem 2rem;
        border-radius: 10px;
        font-size: 1rem;
        font-weight: 600;
        transition: 0.3s ease;
    }
    div.stButton > button:hover {
        transform: scale(1.05);
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }

    /* Prediction box */
    .result-box {
        background: rgba(255, 255, 255, 0.2);
        border-radius: 16px;
        padding: 20px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        color: #fff;
        margin-top: 25px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }

    </style>
""", unsafe_allow_html=True)

# ---- LOAD MODEL ----
with open('30-diamond_model_complete.pkl', 'rb') as f:
    saved_data = pickle.load(f)
    model = saved_data['model']
    encoders = saved_data['encoders']
    scaler = saved_data['scaler']

# ---- TITLE ----
st.markdown("<h2>💎 Diamond Price Predictor</h2>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:#f0f0f0;'>Predict the price of a diamond using its characteristics</p>", unsafe_allow_html=True)

# ---- INPUTS ----
with st.container():
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

# ---- PREDICT ----
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

    st.markdown(f"<div class='result-box'>💰 Estimated Price: ${prediction:,.2f}</div>", unsafe_allow_html=True)
