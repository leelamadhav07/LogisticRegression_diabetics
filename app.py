import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# PAGE CONFIG
st.set_page_config(page_title="Diabetes Prediction App", layout="centered")

st.title("Diabetes Prediction System")
st.write("Logistic Regression Machine Learning Model")

# LOAD DATASET
data = pd.read_csv("diabetes.csv")

# SHOW DATASET
st.subheader("Dataset")

if st.checkbox("Show Dataset"):
    st.write(data.head())

# DATA CLEANING
columns_to_replace = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

for col in columns_to_replace:
    data[col] = data[col].replace(0, data[col].median())

# FEATURES & TARGET
X = data.drop("Outcome", axis=1)
y = data["Outcome"]

# TRAIN TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# FEATURE SCALING
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# MODEL TRAINING
model = LogisticRegression()

model.fit(X_train, y_train)

# MODEL EVALUATION
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

# SHOW ACCURACY
st.subheader("Model Accuracy")

st.write(f"Training Accuracy: {train_accuracy:.2f}")
st.write(f"Testing Accuracy: {test_accuracy:.2f}")

# CONFUSION MATRIX
st.subheader("Confusion Matrix")

cm = confusion_matrix(y_test, y_test_pred)

fig, ax = plt.subplots()

ax.imshow(cm)

for i in range(len(cm)):
    for j in range(len(cm[0])):
        ax.text(j, i, cm[i, j], ha="center", va="center")

ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")

st.pyplot(fig)

# USER INPUT
st.subheader("Predict Diabetes")

pregnancies = st.number_input("Pregnancies", min_value=0)
glucose = st.number_input("Glucose", min_value=0)
blood_pressure = st.number_input("Blood Pressure", min_value=0)
skin_thickness = st.number_input("Skin Thickness", min_value=0)
insulin = st.number_input("Insulin", min_value=0)
bmi = st.number_input("BMI", min_value=0.0)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0)
age = st.number_input("Age", min_value=1)

# PREDICTION BUTTON
if st.button("Predict"):
    input_data = np.array(
        [[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]]
    )

    input_data = scaler.transform(input_data)

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("Person is likely Diabetic")
    else:
        st.success("Person is likely Non-Diabetic")
