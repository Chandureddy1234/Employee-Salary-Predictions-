import streamlit as st
import pandas as pd
import joblib  # Needed to load the model

# Load the trained model
model = joblib.load("best_model.pkl")

st.set_page_config(page_title="Employee Salary Prediction", page_icon="🧑‍💼", layout="centered")
st.title("🧑‍💼 Employee Salary Prediction App")
st.markdown("Use this app to predict whether an employee earns >50K or <=50K based on input features.")

# Sidebar for user input
st.sidebar.header("Enter Employee Details:")

age = st.sidebar.slider("Age", 18, 65, 30)
education = st.sidebar.selectbox("Education Level", ["Bachelors", "Masters", "PhD", "HS-grad", "Assoc", "Some-college"])
occupation = st.sidebar.selectbox("Job Role", [
    "Tech-support", "Craft-repair", "Other-service", "Sales", "Exec-managerial",
    "Prof-specialty", "Handlers-cleaners", "Machine-op-inspect", "Adm-clerical",
    "Farming-fishing", "Transport-moving", "Priv-house-serv", "Protective-serv", "Armed-Forces"
])
hours_per_week = st.sidebar.slider("Hours per Week", 1, 80, 40)
experience = st.sidebar.slider("Years of Experience", 0, 40, 5)

# Prepare input DataFrame (must match model training format)
input_df = pd.DataFrame({
    "age": [age],
    "education": [education],
    "occupation": [occupation],
    "hours-per-week": [hours_per_week],
    "experience": [experience],
})

st.write("### 🧾 Input Data")
st.dataframe(input_df)

# Predict button
if st.button("🔍 Predict Salary Class"):
    prediction = model.predict(input_df)
    st.success(f"💼 Predicted Salary Class: {prediction[0]}")

# Batch Prediction Section
st.markdown("---")
st.markdown("## 📂 Batch Prediction from CSV")
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

if uploaded_file is not None:
    batch_data = pd.read_csv(uploaded_file)
    st.write("📋 Uploaded Data Preview")
    st.dataframe(batch_data.head())
    
    # Make predictions
    batch_preds = model.predict(batch_data)
    batch_data["PredictedClass"] = batch_preds
    
    st.write("🧠 Predictions:")
    st.dataframe(batch_data.head())
    
    csv = batch_data.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download Predictions CSV", data=csv, file_name='predicted_classes.csv', mime='text/csv')
