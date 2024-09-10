import streamlit as st
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np

# Teams and Cities
teams = ['Sunrisers Hyderabad', 'Mumbai Indians', 'Royal Challengers Bangalore',
         'Kolkata Knight Riders', 'Punjab Kings', 'Chennai Super Kings',
         'Rajasthan Royals', 'Delhi Capitals']

cities = ['Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
          'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
          'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
          'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
          'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
          'Sharjah', 'Mohali', 'Bengaluru']

# Load Model, Encoder, Scaler, and Dictionaries
model = pickle.load(open('ipl_model.pkl', 'rb'))
encoder = pickle.load(open('encoder.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))  # Load the trained scaler
dict1 = pickle.load(open('dictionary1.pkl', 'rb'))
dict2 = pickle.load(open('dictionary2.pkl', 'rb'))
dict3 = pickle.load(open('dictionary3.pkl', 'rb'))

st.title("IPL WIN PREDICTOR")

# Input Fields
col1, col2 = st.columns(2)
with col1:
    batting_team = st.selectbox('Select the batting team', sorted(teams))
with col2:
    bowling_team = st.selectbox('Select the bowling team', sorted(teams))

selected_city = st.selectbox('Select host city', sorted(cities))
target = st.number_input('Target')

col3, col4, col5 = st.columns(3)
with col3:
    score = st.number_input('Score')
with col4:
    overs = st.number_input('Overs completed')
with col5:
    wickets = st.number_input('Wickets out')

if st.button('Predict Probability'):
    # Calculating features
    runs_left = target - score
    balls_left = 120 - (overs * 6)
    wickets_remaining = 10 - wickets
    crr = score / overs
    rrr = (runs_left * 6) / balls_left

    # Create input DataFrame
    input_df = pd.DataFrame({
        'batting_team': [dict1.get(batting_team, 'Unknown Team')],
        'bowling_team': [dict2.get(bowling_team, 'Unknown Team')],
        'city': [dict3.get(selected_city, 'Unknown City')],
        'runs_left': [runs_left],
        'balls_left': [balls_left],
        'wickets_left': [wickets_remaining],
        'total_runs_x': [target],
        'crr': [crr],
        'rr': [rrr]
    })

    st.header("Input DataFrame:")
    st.write(input_df)

    # Scaling only numerical columns using the pre-fitted scaler
    numerical_cols = ['batting_team','bowling_team','city','runs_left', 'balls_left', 'wickets_left', 'total_runs_x', 'crr', 'rr']
    input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])  # Use transform, not fit_transform

    st.header("Scaled Input DataFrame:")
    st.write(input_df)

    # Make the prediction
    result = model.predict(input_df)

    win = result[0][0]

    # Display result
    st.header(batting_team + " Win Probability: " + str(np.round(win * 100, 2)) + "%")
