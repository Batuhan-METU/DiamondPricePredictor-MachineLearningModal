import streamlit as st
import pickle
import pandas as pd

# Load model
with open('30-diamond_model_complete.pkl', 'rb') as f:
    saved_data = pickle.load(f)
    model = saved_data['model']
    encoders = saved_data['encoders']
    scaler = saved_data['scaler']

st.title("💎 Diamond Price Predictor")

# User inputs
carat = st.number_input("Carat", 0.1, 5.0, 1.0)
cut = st.selectbox("Cut", ["Fair", "Good", "Very Good", "Premium", "Ideal"])
color = st.selectbox("Color", ["D", "E", "F", "G", "H", "I", "J"])
clarity = st.selectbox("Clarity", ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"])
depth = st.number_input("Depth", 55.0, 70.0, 61.0)
table = st.number_input("Table", 50.0, 70.0, 57.0)
x = st.number_input("x (mm)", 3.0, 10.0, 5.0)
y = st.number_input("y (mm)", 3.0, 10.0, 5.0)
z = st.number_input("z (mm)", 2.0, 10.0, 3.0)

if st.button("Predict Price"):
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
    st.success(f"Estimated Price: ${prediction:,.2f}")
