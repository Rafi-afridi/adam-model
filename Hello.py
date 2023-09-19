import streamlit as st
import pandas as pd
import numpy as np
import time
import joblib

def predict_value(value, classification=True):
    
    if classification is True:
        model_filename = 'trained_model_large.joblib'
    else:
        model_filename = 'trained_model_reg.joblib'
    pipeline = joblib.load(model_filename)
    # Predict the target values on the test data using the fitted pipeline
    
    if classification is True:
        y_pred = pipeline.predict(value).tolist()[0], round(pipeline.predict_proba(value).tolist()[0][0]*100, 2)
    else:
        y_pred = round(pipeline.predict(value).tolist()[0]), 0
    return y_pred
    
import json
with open("feature.json", 'r') as f:
    preprocessing_steps = json.load(f)
    
# Define the input features
input_features = [
    "is_new_user",
    "charter_type",
    "destination_flexible",
    "adults",
    "kids",
    "flexible_date",
    "Request_Month",
    "Request_Day",
    "trip_duration",
    "country_name_in_top_20",
    "total_requests_dest",
    "Destination_in_top_20",
    "total_requests_for_boat",
    "boat_model_name_in_top_20",
    "charter_in_top_5",
    "monthly_average_requests",
    "daily_average_requests",
    "boat_monthly_average_requests",
    "country_monthly_average_requests",
    "request_date_day",
    "hour_request",
    "day_time_request",
    "days_before_departure",
    "in_europe",
    "num_passengers",
    "kid_on_board",
    "civility",
    "user_platform_age_year",
    "is_mac",
    "nb_of_logs",
    "seniority_of_client",
    "boat_bookings_request_ratio",
    "destination_bookings_request_ratio",
    "country_bookings_request_ratio"
]

def function():
    return 1

# Create a sidebar with dropdowns and sliders
st.sidebar.title("Feature Selection")
categorical_features = ['country_name_in_top_20', 'Destination_in_top_20', 'boat_model_name_in_top_20', 'civility']

is_new_user = st.sidebar.selectbox("User is New or Not?", [1, 0])
charter_type = st.sidebar.selectbox("charter_type?", [0, 1, 2, 3, 4, 5])
destination_flexible = st.sidebar.selectbox("Destination is Flexible", [1, 0])
adults = st.sidebar.selectbox("No of Adults", [i for i in range(0, 20)])
kids = st.sidebar.selectbox("No of Kids", [i for i in range(0, 20)])
flexible_date = st.sidebar.selectbox("Date is Flexible or not?", [1, 0])
Request_Month = st.sidebar.selectbox("Request Month", [i for i in range(1, 13)])
Request_Day = st.sidebar.selectbox(f"Request Day", [i for i in range(1, 32)])
trip_duration = st.sidebar.slider("Select trip_duration", 0, 1000, 50)
country_name_in_top_20 = st.sidebar.selectbox(f"Country Name", preprocessing_steps['country_name_in_top_20']+["Other"])
total_requests_dest = st.sidebar.slider("Select total_requests_dest", 0, 1000, 50)
Destination_in_top_20 = st.sidebar.selectbox(f"Destination Name", preprocessing_steps['Destination_in_top_20']+["Other"])
total_requests_for_boat = st.sidebar.slider("Select total_requests_for_boat", 0, 1000, 50)
boat_model_name_in_top_20 = st.sidebar.selectbox(f"Boat Name", preprocessing_steps['boat_model_name_in_top_20']+["Other"])
charter_in_top_5 = st.sidebar.selectbox(f"Charter", preprocessing_steps['charter_in_top_5']+["Other"])
monthly_average_requests = st.sidebar.slider("Select monthly_average_requests", 0, 1000, 50)
daily_average_requests = st.sidebar.slider("Select daily_average_requests", 0, 1000, 50)
boat_monthly_average_requests = st.sidebar.slider("Select boat_monthly_average_requests", 0, 1000, 50)
country_monthly_average_requests = st.sidebar.slider("Select country_monthly_average_requests", 0, 1000, 50)
request_date_day = st.sidebar.selectbox("Request Day Name", ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"])
hour_request = st.sidebar.selectbox("hour_request", [i for i in range(0, 24)])
day_time_request = st.sidebar.selectbox("day_time_request", ["evening", "afternoon", "morning", "night"])
days_before_departure = st.sidebar.slider("Select days_before_departure", 0, 100, 1)
in_europe = st.sidebar.selectbox("in_europe", [1, 0])
num_passengers = st.sidebar.selectbox("num_passengers", [i for i in range(0, 30)])
kid_on_board = 1 if kids > 0 else 0
civility = st.sidebar.selectbox("civility", ["Mr", "Ms", "Dr.", "Prof."])
user_platform_age_year = st.sidebar.selectbox("user_platform_age_year", [i for i in range(0, 30)])
is_mac = st.sidebar.selectbox("is_mac", [1, 0])
nb_of_logs = st.sidebar.slider("Select nb_of_logs", 0, 100, 1)
seniority_of_client = st.sidebar.selectbox("seniority_of_client", [i for i in range(0, 30)])
boat_bookings_request_ratio = st.sidebar.slider("Select boat_bookings_request_ratio", 0, 1000, 10)
destination_bookings_request_ratio = st.sidebar.slider("Select destination_bookings_request_ratio", 0, 1000, 10)
country_bookings_request_ratio = st.sidebar.slider("Select country_bookings_request_ratio", 0, 1000, 10)


data = [[is_new_user,charter_type,destination_flexible, adults, kids, flexible_date, 
         Request_Month,Request_Day,trip_duration,country_name_in_top_20,total_requests_dest,
         Destination_in_top_20,total_requests_for_boat,boat_model_name_in_top_20,charter_in_top_5,
         monthly_average_requests,daily_average_requests,boat_monthly_average_requests,
         country_monthly_average_requests,request_date_day,hour_request,day_time_request,
         days_before_departure,in_europe,num_passengers,kid_on_board,civility,
         user_platform_age_year,is_mac,nb_of_logs,seniority_of_client,boat_bookings_request_ratio/100,
         destination_bookings_request_ratio/100,country_bookings_request_ratio/100]]
         
value = pd.DataFrame(data=data, columns=input_features)

# Create a button to trigger predictions
if st.sidebar.button("Predict"):
    # Simulate a loading process
    with st.spinner("Predicting..."):
        time.sleep(2)

    # Display the prediction results
    st.success("Prediction Complete!")
    st.write("Predicted Probabilities:")
    label, probablity = predict_value(value, classification=True)
    probabilities = {"Probability of Booking Ship" : f"{100-probablity}%"}
    st.write(probabilities)
