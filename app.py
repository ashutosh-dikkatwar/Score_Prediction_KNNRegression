
import pickle
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="wide"
)


@st.cache_resource
def load_model():
    with open("KNN_regression.pkl", "rb") as file:
        return pickle.load(file)


model = load_model()

st.title("🎓 Student Performance Prediction")
st.write("Enter student details to predict the result.")

st.divider()

# Categorical value mappings
# IMPORTANT: Confirm these codes against your training data.
gender_map = {
    "Female": 0,
    "Male": 1
}

course_map = {
    "Arts": 0,
    "Commerce": 1,
    "Science": 2
}

internet_map = {
    "No": 0,
    "Yes": 1
}

sleep_quality_map = {
    "Average": 0,
    "Good": 1,
    "Poor": 2
}

study_method_map = {
    "Coaching": 0,
    "Group Study": 1,
    "Online Videos": 2,
    "Self Study": 3
}

facility_map = {
    "High": 0,
    "Low": 1,
    "Medium": 2
}

exam_difficulty_map = {
    "Easy": 0,
    "Hard": 1,
    "Moderate": 2
}

# Input form
with st.form("student_form"):

    st.subheader("Personal Information")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input(
            "Age",
            min_value=1,
            max_value=100,
            value=20
        )

        gender = st.selectbox(
            "Gender",
            list(gender_map.keys())
        )

    with col2:
        course = st.selectbox(
            "Course",
            list(course_map.keys())
        )

        internet_access = st.selectbox(
            "Internet Access",
            list(internet_map.keys())
        )

    st.subheader("Study Information")

    col3, col4 = st.columns(2)

    with col3:
        study_hours = st.number_input(
            "Study Hours",
            min_value=0.0,
            max_value=24.0,
            value=3.0,
            step=0.5
        )

        attendance = st.number_input(
            "Class Attendance (%)",
            min_value=0.0,
            max_value=100.0,
            value=75.0,
            step=1.0
        )

        sleep_hours = st.number_input(
            "Sleep Hours",
            min_value=0.0,
            max_value=24.0,
            value=7.0,
            step=0.5
        )

    with col4:
        sleep_quality = st.selectbox(
            "Sleep Quality",
            list(sleep_quality_map.keys())
        )

        study_method = st.selectbox(
            "Study Method",
            list(study_method_map.keys())
        )

    st.subheader("Exam and Facility Information")

    col5, col6 = st.columns(2)

    with col5:
        facility_rating = st.selectbox(
            "Facility Rating",
            list(facility_map.keys())
        )

    with col6:
        exam_difficulty = st.selectbox(
            "Exam Difficulty",
            list(exam_difficulty_map.keys())
        )

    submitted = st.form_submit_button(
        "Predict Student Result",
        type="primary",
        use_container_width=True
    )


if submitted:

    # Convert selected text values to numeric codes
    input_data = pd.DataFrame([{
        "age": age,
        "gender": gender_map[gender],
        "course": course_map[course],
        "study_hours": study_hours,
        "class_attendance": attendance,
        "internet_access": internet_map[internet_access],
        "sleep_hours": sleep_hours,
        "sleep_quality": sleep_quality_map[sleep_quality],
        "study_method": study_method_map[study_method],
        "facility_rating": facility_map[facility_rating],
        "exam_difficulty": exam_difficulty_map[exam_difficulty]
    }])

    try:
        prediction = model.predict(input_data)[0]

        st.divider()
        st.subheader("Prediction Result")

        st.success("Prediction completed successfully!")

        st.metric(
            label="Predicted Student Result",
            value=f"{prediction:.2f}"
        )

        with st.expander("View input data sent to the model"):
            st.dataframe(input_data, use_container_width=True)

    except Exception as error:
        st.error(f"Prediction failed: {error}")
