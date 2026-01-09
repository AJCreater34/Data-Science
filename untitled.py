import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression

# ----------------------
# TRAIN THE MODEL
# ----------------------

data = {
    "attendance": [90, 75, 60, 85, 50],
    "homework_done": [1, 1, 0, 1, 0],
    "test_score": [88, 72, 45, 80, 40],
    "final_result": [1, 1, 0, 1, 0]  # Pass=1, Fail=0
}

df = pd.DataFrame(data)

X = df[["attendance", "homework_done", "test_score"]]
y = df["final_result"]

model = LogisticRegression()
model.fit(X, y)

# ----------------------
# APP UI
# ----------------------

st.title("Student Performance Predictor")

attendance = st.slider("Attendance (%)", 0, 100, 50)
homework = st.selectbox("Homework Done?", ["Yes", "No"])
test_score = st.slider("Test Score: ", 0, 100, 50)

homework_value = 1 if homework == "Yes" else 0

# ----------------------
# WEIGHTS
# ----------------------

weights = model.coef_[0]
features = X.columns

explanation = []

if homework_value == 0:
    explanation.append("Homework was not done")
    
if attendance < 60:
    explanation.append("Attendence is low")
    
if test_score < 50:
    explanation.append("Test Score is low")

# ----------------------
# PREDICTION
# ----------------------

if st.button("Predict"):
    input_data = [[attendance, homework_value, test_score]]
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("Prediction: you will Pass!")
        st.success("Good Job! Keep working hard!")
        
    else:
        st.error("Prediction: Fail")
        st.write("Why did you get this result: ")
        for reason in explanation:
            st.write("-", reason)






