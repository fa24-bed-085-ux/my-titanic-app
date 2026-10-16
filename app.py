import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('svc_trained_model.pkl')

st.title('Titanic Survival Prediction App')
st.write('Enter your details below to check the survival prediction!')

# Input fields based on your model features: PClass, Gender, Sibling, Embarked
pclass = st.selectbox('Passenger Class (PClass)', [1, 2, 3])
gender = st.selectbox('Gender (0 = Female, 1 = Male)', [0, 1])
sibling = st.number_input('Number of Siblings / Spouses', min_value=0, max_value=10, value=0)
embarked = st.selectbox('Embarked Port (0 = C, 1 = Q, 2 = S)', [0, 1, 2])

# Prediction button
if st.button('Predict'):
    # Convert input data into a DataFrame
    input_data = pd.DataFrame([[pclass, gender, sibling, embarked]], 
                              columns=['PClass', 'Gender', 'Sibling', 'Embarked'])
    
    # Make prediction using the model
    prediction = model.predict(input_data)
    
    # Show the result
    if prediction[0] == 1:
        st.success('Congratulations! The passenger has a chance of survival. ')
    else:
        st.warning('The passenger has a lower chance of survival. ')
