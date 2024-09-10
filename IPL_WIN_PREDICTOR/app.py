import streamlit as st
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np

teams = ['Sunrisers Hyderabad',
 'Mumbai Indians',
 'Royal Challengers Bangalore',
 'Kolkata Knight Riders',
 'Punjab Kings',
 'Chennai Super Kings',
 'Rajasthan Royals',
 'Delhi Capitals']

cities = ['Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
       'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
       'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
       'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
       'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
       'Sharjah', 'Mohali', 'Bengaluru']

model = pickle.load(open('ipl_model.pkl','rb'))
encoder = pickle.load(open('encoder.pkl','rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
dict1 = pickle.load(open('dictionary1.pkl','rb'))
dict2 = pickle.load(open('dictionary2.pkl','rb'))
dict3 = pickle.load(open('dictionary3.pkl','rb'))

st.title("IPL WIN PREDICTOR")

col1, col2 = st.columns(2)

with col1:
    batting_team = st.selectbox('Select the batting team',sorted(teams))
with col2:
    bowling_team = st.selectbox('Select the bowling team',sorted(teams))

selected_city = st.selectbox('Select host city',sorted(cities))

target = st.number_input('Target')

col3,col4,col5 = st.columns(3)

with col3:
    score = st.number_input('Score')
with col4:
    overs = st.number_input('Overs completed')
with col5:
    wickets = st.number_input('Wickets out')

if st.button('Predict Probability'):
    runs_left = target - score
    balls_left = 120 - (overs*6)
    wickets = 10 - wickets
    crr = score/overs
    rrr = (runs_left*6)/balls_left

    batting_team_value = dict1.get(batting_team, 'Unknown Team')  # Handle cases where key doesn't exist


    input_df = pd.DataFrame({'batting_team':[dict1[batting_team]],'bowling_team':[dict2[bowling_team]],'city':[dict3[selected_city]],'runs_left':[runs_left],'balls_left':[balls_left],'wickets_left':[wickets],'total_runs_x':[target],'crr':[crr],'rr':[rrr]})





    numerical_cols = input_df.select_dtypes(include=['float64', 'int64']).columns

    input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])
    

    result = model.predict(input_df)

    #loss = result[0][0]
    win = result[0][0]

    st.header(batting_team + "- " + str(np.round(win*100)) + "%")
    #st.header(bowling_team + "- " + str(round(loss*100)) + "%")