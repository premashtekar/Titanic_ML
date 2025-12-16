import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="centered"
)

# ---------------- DARK BLUE THEME ----------------
st.markdown("""
<style>
body {
    background-color: #020b1c;
}
.main {
    background-color: #020b1c;
}
h1, h2, h3 {
    color: #7ec8ff;
}
label {
    color: white !important;
}
.stButton > button {
    background-color: #0b5ed7;
    color: white;
    border-radius: 10px;
    font-size: 18px;
    padding: 10px;
}
.stButton > button:hover {
    background-color: #0dcaf0;
    color: black;
}
.success {
    color: #00ffcc;
}
.error {
    color: #ff4b4b;
}
</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.markdown("<h1 style='text-align:center;'>🚢 Titanic Survival Predictor</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align:center; color:white;'>Machine Learning + Streamlit</h4>", unsafe_allow_html=True)
st.write("")

# ---------------- MODEL TRAINING FUNCTION ----------------
@st.cache_data
def train_model():
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")

    features = ["Pclass", "Sex", "Age", "Fare"]

    # Handle missing values
    train["Age"].fillna(train["Age"].median(), inplace=True)
    test["Age"].fillna(test["Age"].median(), inplace=True)
    test["Fare"].fillna(test["Fare"].median(), inplace=True)

    # Encode sex
    train["Sex"] = train["Sex"].map({"male": 0, "female": 1})
    test["Sex"] = test["Sex"].map({"male": 0, "female": 1})

    X = train[features]
    y = train["Survived"]
    X_test = test[features]

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_test = scaler.transform(X_test)

    model = LogisticRegression()
    model.fit(X, y)

    # Save model
    pickle.dump(model, open("model.pkl", "wb"))
    pickle.dump(scaler, open("scaler.pkl", "wb"))

    # Kaggle submission
    preds = model.predict(X_test)
    submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": preds
    })
    submission.to_csv("submission.csv", index=False)

    return model, scaler

# ---------------- LOAD OR TRAIN MODEL ----------------
if not os.path.exists("model.pkl"):
    model, scaler = train_model()
else:
    model = pickle.load(open("model.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))

# ---------------- USER INPUT UI ----------------
pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.selectbox("Gender", ["Male", "Female"])
age = st.slider("Age", 1, 80, 25)
fare = st.number_input("Fare (Ticket Price)", min_value=0.0, value=30.0)

sex = 1 if sex == "Female" else 0

input_data = np.array([[pclass, sex, age, fare]])
input_data = scaler.transform(input_data)

# ---------------- PREDICTION ----------------
if st.button("Predict Survival"):
    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.success("✅ Passenger is LIKELY TO SURVIVE")
    else:
        st.error("❌ Passenger is UNLIKELY TO SURVIVE")

# ---------------- FOOTER ----------------
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center; color:#7ec8ff;'>Single-file ML App | Titanic Dataset</p>",
    unsafe_allow_html=True
)
