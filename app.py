import pickle
import pandas as pd
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Titanic Survival Prediction", page_icon="🚢", layout="centered"
)


# Load the trained SVC model
@st.cache_resource
def load_model():
  with open("svc_trained_model.pkl", "rb") as file:
    model = pickle.load(file)
  return model


model = load_model()

st.title("🚢 Titanic Passenger Survival Prediction App")
st.write(
    "Enter the passenger details below to check if they would have survived"
    " the Titanic disaster!"
)

# Input fields for user features (adjust fields according to your model's training features)
pclass = st.selectbox("Passenger Class (Pclass)", [1, 2, 3])
sex = st.selectbox("Sex", ["Male", "Female"])
age = st.slider("Age", 0, 80, 25)
sibsp = st.number_input(
    "Siblings/Spouses Aboard (SibSp)", min_value=0, max_value=10, value=0
)
parch = st.number_input(
    "Parents/Children Aboard (Parch)", min_value=0, max_value=10, value=0
)
fare = st.number_input("Fare", min_value=0.0, max_value=600.0, value=32.2)
embarked = st.selectbox("Port of Embarkation", ["C", "Q", "S"])

# Convert categorical inputs to numerical format if your model expects numbers
sex_val = 0 if sex == "Male" else 1
embarked_map = {"C": 0, "Q": 1, "S": 2}
embarked_val = embarked_map[embarked]

# Create input dataframe for prediction
input_data = pd.DataFrame(
    [[pclass, sex_val, age, sibsp, parch, fare, embarked_val]],
    columns=[
        "Pclass",
        "Sex",
        "Age",
        "SibSp",
        "Parch",
        "Fare",
        "Embarked",
    ],  # Column names must match your training data features
)

# Prediction button
if st.button("Predict Survival"):
  prediction = model.predict(input_data)

  if prediction[0] == 1:
    st.success("🎉 Congratulations! This passenger would have survived.")
  else:
    st.error("⚠️ Unfortunately, this passenger would not have survived.")
