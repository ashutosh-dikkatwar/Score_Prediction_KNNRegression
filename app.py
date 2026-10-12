import pickle
import pandas as pd
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="KNN Regression Predictor",
    page_icon="📊",
    layout="centered"
)

# Load trained model
@st.cache_resource
def load_model():
    with open("KNN_regression.pkl", "rb") as file:
        model = pickle.load(file)
    return model

model = load_model()

# Application title
st.title("📊 KNN Regression Predictor")
st.write("Enter the details below to predict the result.")

st.divider()

# Input fields
st.subheader("Enter Input Details")

age = st.number_input("Age", min_value=0, max_value=100, value=20)

gender = st.selectbox("Gender (encoded)", [0, 1])

course = st.number_input(
    "Course (encoded)", min_value=0, value=0
)

study_hours = st.number_input(
    "Study Hours", min_value=0.0, value=3.0, step=0.5
)

attendance = st.number_input(
    "Attendance (%)",
    min_value=0.0,
    max_value=100.0,
    value=75.0
)

internet_access = st.selectbox(
    "Internet Access (encoded)", [0, 1]
)

sleep_hours = st.number_input(
    "Sleep Hours", min_value=0.0, value=7.0, step=0.5
)

sleep_quality = st.selectbox(
    "Sleep Quality (encoded)", [0, 1, 2]
)

study_method = st.number_input(
    "Study Method (encoded)", min_value=0, value=0
)

facility_rating = st.selectbox(
    "Facility Rating (encoded)", [0, 1, 2]
)

exam_difficulty = st.selectbox(
    "Exam Difficulty (encoded)", [0, 1, 2]
)

# Prediction button
if st.button("Predict", type="primary"):

    # Keep feature names and order consistent
    # with the model's training data
    input_data = pd.DataFrame([{
        "age": age,
        "gender": gender,
        "course": course,
        "study_hours": study_hours,
        "class_attendance": attendance,
        "internet_access": internet_access,
        "sleep_hours": sleep_hours,
        "sleep_quality": sleep_quality,
        "study_method": study_method,
        "facility_rating": facility_rating,
        "exam_difficulty": exam_difficulty
    }])

    try:
        prediction = model.predict(input_data)

        st.success("Prediction completed successfully!")
        st.metric(
            label="Predicted Value",
            value=f"{prediction[0]:.2f}"
        )

    except Exception as e:
        st.error(f"Prediction error: {e}")
        st.warning(
            "Check the feature names, feature order, "
            "and categorical encoding used during training."
        )
