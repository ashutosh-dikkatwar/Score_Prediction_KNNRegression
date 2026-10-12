
import pickle
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Student Performance AI",
    page_icon="🎓",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #f5f7ff, #eef4ff);
}
.block-container {
    max-width: 1200px;
    padding-top: 2rem;
}
.hero {
    background: linear-gradient(120deg, #172554, #2563eb, #6366f1);
    padding: 30px;
    border-radius: 20px;
    color: white;
    margin-bottom: 25px;
}
.hero h1 {
    color: white;
    font-size: 2.2rem;
}
.hero p {
    color: #e0e7ff;
}
.result-card {
    background: #ecfdf5;
    border: 1px solid #bbf7d0;
    padding: 25px;
    border-radius: 18px;
    margin-top: 20px;
}
.result-value {
    color: #166534;
    font-size: 2.8rem;
    font-weight: 800;
}
[data-testid="stMetric"] {
    background: white;
    border: 1px solid #e2e8f0;
    padding: 15px;
    border-radius: 14px;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    with open("KNN_regression.pkl", "rb") as f:
        return pickle.load(f)


try:
    model = load_model()
except FileNotFoundError:
    st.error("Please keep KNN_regression.pkl in the same folder as app.py.")
    st.stop()


# IMPORTANT: Replace these example mappings with your
# actual categorical encodings used during model training.
gender_map = {"Female": 0, "Male": 1}
course_map = {"Arts": 0, "Commerce": 1, "Science": 2}
internet_map = {"No": 0, "Yes": 1}

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


# Sidebar
with st.sidebar:
    st.title("🎓 Student AI")
    st.caption("KNN Regression Dashboard")
    st.divider()
    st.markdown("### How to use")
    st.write("1. Enter student details.")
    st.write("2. Select study habits.")
    st.write("3. Click Predict.")
    st.divider()
    st.caption("Powered by Python and Streamlit")


# Main header
st.markdown("""
<div class="hero">
    <h1>🎓 Student Performance Predictor</h1>
    <p>Enter student details and explore your machine learning prediction.</p>
</div>
""", unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)
c1.metric("ML Algorithm", "KNN Regression")
c2.metric("Input Features", "11")
c3.metric("Output", "Prediction")


# Input form
with st.form("prediction_form"):

    st.subheader("👤 Personal Details")

    c1, c2, c3 = st.columns(3)

    with c1:
        age = st.number_input(
            "Age", min_value=1, max_value=100, value=20
        )

    with c2:
        gender = st.selectbox(
            "Gender", list(gender_map.keys())
        )

    with c3:
        course = st.selectbox(
            "Course", list(course_map.keys())
        )

    st.divider()
    st.subheader("📚 Study Habits")

    c1, c2, c3 = st.columns(3)

    with c1:
        study_hours = st.number_input(
            "Study Hours Per Day",
            min_value=0.0, max_value=24.0,
            value=3.0, step=0.5
        )

        study_method = st.selectbox(
            "Study Method", list(study_method_map.keys())
        )

    with c2:
        attendance = st.slider(
            "Attendance (%)", 0.0, 100.0, 75.0
        )

        internet_access = st.selectbox(
            "Internet Access", list(internet_map.keys())
        )

    with c3:
        sleep_hours = st.number_input(
            "Sleep Hours Per Day",
            min_value=0.0, max_value=24.0,
            value=7.0, step=0.5
        )

        sleep_quality = st.selectbox(
            "Sleep Quality", list(sleep_quality_map.keys())
        )

    st.divider()
    st.subheader("📝 Exam and Facilities")

    c1, c2 = st.columns(2)

    with c1:
        facility_rating = st.selectbox(
            "Facility Rating", list(facility_map.keys())
        )

    with c2:
        exam_difficulty = st.selectbox(
            "Exam Difficulty", list(exam_difficulty_map.keys())
        )

    submitted = st.form_submit_button(
        "✨ Predict Student Result",
        type="primary",
        use_container_width=True
    )


# Prediction
if submitted:

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

        st.markdown(f"""
        <div class="result-card">
            <h3>✅ Prediction Result</h3>
            <div class="result-value">{prediction:.2f}</div>
            <p>Predicted value from your KNN regression model</p>
        </div>
        """, unsafe_allow_html=True)

        st.subheader("📋 Student Summary")

        c1, c2, c3 = st.columns(3)
        c1.metric("Age", str(age))
        c2.metric("Study Hours", f"{study_hours:g} hrs/day")
        c3.metric("Attendance", f"{attendance:g}%")

        with st.expander("View model input values"):
            st.dataframe(input_data, use_container_width=True)

        st.info(
            "The result is a model prediction, not a guarantee "
            "of actual student performance."
        )

    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.warning(
            "Check that feature names, feature order, and categorical "
            "encodings match your original training data."
        )
