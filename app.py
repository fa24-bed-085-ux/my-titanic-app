import streamlit as st
import pandas as pd
import joblib

# Model load karna (apni model file ka naam yahan 'model.pkl' rakhein)
model = joblib.load('model.pkl')

st.title('Titanic Survival Prediction App')
st.write('Apni details niche darj karein aur check karein ki prediction kya aati hai!')

# Input fields based on your model features: PClass, Gender, Sibling, Embarked
pclass = st.selectbox('Passenger Class (PClass)', [1, 2, 3])
gender = st.selectbox('Gender (0 = Female, 1 = Male)', [0, 1])
sibling = st.number_input('Number of Siblings / Spouses', min_value=0, max_value=10, value=0)
embarked = st.selectbox('Embarked Port (0 = C, 1 = Q, 2 = S)', [0, 1, 2])

# Prediction button
if st.button('Predict'):
    # Input data ko DataFrame mein convert karna
    input_data = pd.DataFrame([[pclass, gender, sibling, embarked]], 
                              columns=['PClass', 'Gender', 'Sibling', 'Embarked'])
    
    # Model se prediction lena
    prediction = model.predict(input_data)
    
    # Result show karna
    if prediction[0] == 1:
        st.success('Mubarak ho! Passenger ke survive karne ka chance hai. 🎉')
    else:
        st.warning('Passenger ke survive karne ka chance kam hai. ⚠️')
