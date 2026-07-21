import gradio as gr
import joblib
import numpy as np

# Load trained model
model = joblib.load("student_dropout_model.pkl")


def predict_dropout(
    age,
    gender,
    year_of_study,
    attendance_percentage,
    study_hours,
    previous_gap,
    backlogs,
    financial_stress,
    stress_level,
    burnout_level,
):
    # Encode gender
    gender = 1 if gender == "Male" else 0

    # Encode burnout level
    burnout_mapping = {
        "Low": 0,
        "Medium": 1,
        "High": 2,
    }
    burnout = burnout_mapping[burnout_level]

    # Create input array
    input_data = np.array([[
        age,
        gender,
        year_of_study,
        attendance_percentage,
        study_hours,
        previous_gap,
        backlogs,
        financial_stress,
        stress_level,
        burnout
    ]])

    # Predict
    prediction = model.predict(input_data)[0]

    if prediction == 1:
        return "Student will drop out."
    else:
        return "Student will not drop out."


# Create Gradio interface
student_dropout_app = gr.Interface(
    fn=predict_dropout,
    inputs=[
        gr.Number(label="Age"),
        gr.Radio(["Male", "Female"], label="Gender"),
        gr.Dropdown([1, 2, 3, 4], label="Year of Study"),
        gr.Slider(0, 100, label="Attendance Percentage"),
        gr.Number(label="Study Hours per Day"),
        gr.Number(label="Previous GPA Gap"),
        gr.Number(label="Backlogs"),
        gr.Slider(1, 10, step=1, label="Financial Stress Score"),
        gr.Slider(1, 10, step=1, label="Stress Level"),
        gr.Dropdown(["Low", "Medium", "High"], label="Burnout Level"),
    ],
    outputs=gr.Textbox(label="Prediction"),
    title="Student Dropout Risk Prediction",
    description="Predict whether a student has a high or low dropout risk.",
)

# Launch app
student_dropout_app.launch()