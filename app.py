import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained model and scaler
model = joblib.load(r'F:\Capston Project\best_model.pkl')
scaler = joblib.load(r'F:\Capston Project\scaler.pkl')
label_encoders = joblib.load(r'F:\Capston Project\label_encoder.pkl')

# Title of the app
st.title("Fraud Detection System")

# User input for transaction details
st.header("Enter Transaction Details")

step = st.number_input("Step", min_value=1, max_value=100, value=1)
transaction_type = st.selectbox("Transaction Type", options=['PAYMENT', 'CASH_OUT', 'CASH_IN', 'DEBIT', 'TRANSFER'])
amount = st.number_input("Transaction Amount", min_value=1, value=100)
bal_before = st.number_input("Balance Before Transaction", min_value=0, value=1000)
bal_after = st.number_input("Balance After Transaction", min_value=0, value=1000)
rec_bal_before = st.number_input("Recipient Balance Before Transaction", min_value=0, value=1000)
rec_bal_after = st.number_input("Recipient Balance After Transaction", min_value=0, value=1000)

# Prepare the input data for prediction
input_data = pd.DataFrame({
    'step': [step],
    'type': [label_encoders['type'].transform([transaction_type])[0]],
    'amount': [amount],
    'bal_before_transaction': [bal_before],
    'bal_after_transaction': [bal_after],
    'bal_of_recepient_before_transaction': [rec_bal_before],
    'bal_of_receipient_after_transaction': [rec_bal_after]
})

# Button for prediction
if st.button("Predict Fraud"):
    # Scale the input data
    input_scaled = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(input_scaled)

    # Display the result
    if prediction[0] == 1:
        st.error("This transaction is predicted to be **Fraudulent**.")
    else:
        st.success("This transaction is predicted to be **Not Fraudulent**.")

# Run the Streamlit app
if __name__ == "__main__":
    st.write("Kalpesh Tarsariya")
