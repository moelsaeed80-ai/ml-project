import streamlit as st
import pickle
import numpy as np

st.title("ML Project 🚀")

# تحميل الموديل والسكيلر
model = pickle.load(open("best_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.write("Enter values:")

# مثال إدخال (غيره حسب الداتا عندك)
x1 = st.number_input("Feature 1")
x2 = st.number_input("Feature 2")

if st.button("Predict"):
    input_data = np.array([[x1, x2]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)

    st.success(f"Prediction: {prediction[0]}")
