import streamlit as st
import pandas as pd
import joblib
import os
print("Files saved successfully!")

# Page Configuration
st.set_page_config(
    page_title="Hotel Booking Cancellation Prediction",
    layout="centered"
)
st.title("Hotel Booking Cancellation Prediction")

# Check Required Files
required_files=["hotel_model.pkl", "scaler.pkl", "columns.pkl"]
for file in required_files:
    if not os.path.exists(file):
        st.error(f"Error: '{file}' not found. Train your model first.")
        st.stop()

# Load Model Files
model=joblib.load("hotel_model.pkl")
scaler=joblib.load("scaler.pkl")
encoded_columns=joblib.load("columns.pkl")

# # Save model
# joblib.dump(model,"hotel_model.pkl")

# # Save scaler
# joblib.dump(scaler,"scaler.pkl")

# # Save encoded column names
# joblib.dump(X.columns.tolist(),"columns.pkl")
# -----------------------------
# User Inputs
# -----------------------------
lead_time=st.number_input("Lead Time",min_value=0,value=10)
adults=st.number_input("Adults",min_value=1,value=2)
children=st.number_input("Children",min_value=0,value=0)
babies=st.number_input("Babies",min_value=0,value=0)
previous_cancellations = st.number_input(
    "Previous Cancellations",
    min_value=0,
    value=0
)
booking_changes = st.number_input(
    "Booking Changes",
    min_value=0,
    value=0
)
required_car_parking_spaces = st.number_input(
    "Required Car Parking Spaces",
    min_value=0,
    value=0
)
total_of_special_requests = st.number_input(
    "Total Special Requests",
    min_value=0,
    value=0
)
hotel = st.selectbox(
    "Hotel",
    ["City Hotel", "Resort Hotel"]
)
meal = st.selectbox(
    "Meal",
    ["BB", "HB", "FB", "SC"]
)
customer_type = st.selectbox(
    "Customer Type",
    ["Transient", "Contract", "Group", "Transient-Party"]
)

# Prediction
if st.button("Predict"):
    input_data = {
        "lead_time": lead_time,
        "adults": adults,
        "children": children,
        "babies": babies,
        "previous_cancellations": previous_cancellations,
        "booking_changes": booking_changes,
        "required_car_parking_spaces": required_car_parking_spaces,
        "total_of_special_requests": total_of_special_requests,
        "hotel": hotel,
        "meal": meal,
        "customer_type": customer_type
    }
    # Create DataFrame
    input_df = pd.DataFrame([input_data])
    # One-Hot Encode
    input_df = pd.get_dummies(input_df)
    # Match training columns
    input_df = input_df.reindex(columns=encoded_columns, fill_value=0)
    # Scale
    input_scaled = scaler.transform(input_df)
    # Predict
    prediction = model.predict(input_scaled)
    if prediction[0] == 1:
        st.error("❌Booking will be Cancelled")
    else:
        st.success("✅Booking will NOT be Cancelled")