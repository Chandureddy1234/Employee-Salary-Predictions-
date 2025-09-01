import streamlit as st
import pandas as pd
import joblib

# Load the trained model pipeline
try:
    pipeline = joblib.load("best_model.pkl")
except FileNotFoundError:
    st.error("Error: The 'best_model.pkl' file was not found. Please run the Jupyter Notebook first to train and save the model.")
    st.stop()
except Exception as e:
    st.error(f"Error loading the model: {e}")
    st.stop()

st.set_page_config(page_title="Employee Salary Prediction", page_icon="🧑‍💼", layout="centered")

st.title("🧑‍💼 Employee Salary Prediction App")
st.write("Predict whether an employee earns >50K or <=50K based on their profile. This app demonstrates a complete data science pipeline from data analysis to model deployment.")

# --- Individual Prediction Section ---
st.header("🔮 Make an Individual Prediction")
st.write("Enter the details of a single employee to get a salary prediction.")

with st.container():
    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Age", 17, 90, 30)
        workclass = st.selectbox("Work Class", ['Private', 'Local-gov', 'Self-emp-not-inc', 'Federal-gov', 'State-gov', 'Self-emp-inc', 'Without-pay', 'Never-worked'])
        education = st.selectbox("Education Level", ['Assoc-voc', 'Bachelors', 'Some-college', '10th', 'HS-grad', 'Masters', '11th', 'Assoc-acdm', 'Prof-school', '7th-8th', '9th', '12th', 'Doctorate', '5th-6th', '1st-4th', 'Preschool'])
        marital_status = st.selectbox("Marital Status", ['Married-civ-spouse', 'Never-married', 'Divorced', 'Separated', 'Widowed', 'Married-spouse-absent', 'Married-AF-spouse'])
        occupation = st.selectbox("Job Role", ['Protective-serv', 'Machine-op-inspct', 'Farming-fishing', 'Other-service', 'Adm-clerical', 'Craft-repair', 'Tech-support', 'Sales', 'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners', 'Transport-moving', 'Priv-house-serv', 'Armed-Forces'])
    
    with col2:
        relationship = st.selectbox("Relationship", ['Wife', 'Own-child', 'Husband', 'Unmarried', 'Not-in-family', 'Other-relative'])
        race = st.selectbox("Race", ['White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other'])
        gender = st.selectbox("Gender", ['Male', 'Female'])
        capital_gain = st.number_input("Capital Gain", value=0, step=100)
        capital_loss = st.number_input("Capital Loss", value=0, step=100)
        hours_per_week = st.slider("Hours per Week", 1, 99, 40)
        native_country = st.selectbox("Native Country", ['United-States', 'Mexico', 'Peru', 'Hungary', 'Columbia', 'China', 'Philippines', 'Ecuador', 'Germany', 'Portugal', 'England', 'Cambodia', 'Jamaica', 'France', 'Poland', 'Dominican-Republic', 'Guatemala', 'Canada', 'Taiwan', 'El-Salvador', 'India', 'Japan', 'Iran', 'Cuba', 'South', 'Puerto-Rico', 'Scotland', 'Nicaragua', 'Haiti', 'Italy', 'Vietnam', 'Honduras', 'Laos', 'Thailand', 'Ireland', 'Outlying-US(Guam-USVI-etc)', 'Greece', 'Trinadad&Tobago', 'Yugoslavia', 'Holand-Netherlands', 'Hong', ''])

    input_data = pd.DataFrame({
        'age': [age],
        'workclass': [workclass],
        'education': [education],
        'marital-status': [marital_status],
        'occupation': [occupation],
        'relationship': [relationship],
        'race': [race],
        'gender': [gender],
        'capital-gain': [capital_gain],
        'capital-loss': [capital_loss],
        'hours-per-week': [hours_per_week],
        'native-country': [native_country]
    })

    st.write("---")
    st.subheader("Your Input Data:")
    st.dataframe(input_data)

    if st.button("🔍 Predict Salary Class"):
        try:
            prediction = pipeline.predict(input_data)[0]
            prediction_text = "💰 >50K" if prediction == 1 else "💼 <=50K"
            st.success(f"Predicted Salary Class: **{prediction_text}**")
        except Exception as e:
            st.error(f"Prediction failed. Error: {e}")

st.markdown('<hr style="border:1px solid #ddd;">', unsafe_allow_html=True)

# --- Batch Prediction Section ---
st.header("📂 Batch Prediction from CSV")
st.write("Upload a CSV file with multiple employee records to get a batch prediction.")
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

if uploaded_file is not None:
    try:
        batch_data = pd.read_csv(uploaded_file)
        
        # Make a copy to avoid SettingWithCopyWarning
        batch_data_copy = batch_data.copy()
        
        # The pipeline handles preprocessing automatically
        batch_preds = pipeline.predict(batch_data_copy)
        batch_data["PredictedClass"] = [" >50K" if p == 1 else " <=50K" for p in batch_preds]
        
        st.write("🧠 Predictions:")
        st.dataframe(batch_data.head())
        
        csv = batch_data.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Download Predictions CSV",
            data=csv,
            file_name='predicted_classes.csv',
            mime='text/csv'
        )
    except Exception as e:
        st.error(f"Batch prediction failed. Ensure the uploaded CSV has the correct columns. Error: {e}")

