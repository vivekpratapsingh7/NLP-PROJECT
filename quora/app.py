# Streamlit app

import streamlit as st
import pickle
import numpy as np

import tensorflow as tf
tf.compat.v1.reset_default_graph()


from helper import query_point_creator

# Load the model and vectorizer

model_path = 'model.pkl'
model = None

try:
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
except Exception as e:
    st.error(f"Error loading the model: {e}")

st.title("Quora Question Pair Similarity Prediction")

q1 = st.text_input("Enter Question 1:")
q2 = st.text_input("Enter Question 2:")

if st.button("Predict"):
    if q1 and q2:
        # Create the feature vector
        query_point = query_point_creator(q1, q2)

        # Predict similarity
        prediction = model.predict(query_point)[0]
        if prediction<0.5:
            prediction = 0
        else:
            prediction = 1
        similarity = "Duplicate" if prediction == 1 else "Not Duplicate"

        st.write(f"The questions are: **{similarity}**")
    else:
        st.write("Please enter both questions to get a prediction.")
