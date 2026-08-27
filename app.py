import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

# --------------------------------
# Page Configuration
# --------------------------------

st.set_page_config(
    page_title="MediVision AI",
    page_icon="🏥",
    layout="wide"
)

# --------------------------------
# Title
# --------------------------------

st.title("🏥 MediVision AI")
st.subheader("AI + ML + DL Smart Health Analyzer")

st.write(
    "Educational healthcare analysis project using "
    "Artificial Intelligence, Machine Learning and Deep Learning."
)

st.divider()

# --------------------------------
# Dataset
# --------------------------------

data = {
    "age": [22, 25, 30, 35, 40, 45, 28, 50, 32, 55],
    "bp": [110, 120, 125, 135, 140, 150, 118, 155, 130, 160],
    "sugar": [90, 95, 100, 115, 125, 140, 92, 150, 105, 160],
    "cholesterol": [170, 180, 190, 210, 220, 240, 175, 250, 200, 260],
    "bmi": [21, 22, 24, 27, 29, 31, 23, 33, 25, 35],
    "risk": [0, 0, 0, 1, 1, 1, 0, 1, 0, 1]
}

df = pd.DataFrame(data)

# --------------------------------
# Features and Target
# --------------------------------

X = df[[
    "age",
    "bp",
    "sugar",
    "cholesterol",
    "bmi"
]]

y = df["risk"]

# --------------------------------
# Train Test Split
# --------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# --------------------------------
# ML Model - Random Forest
# --------------------------------

ml_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

ml_model.fit(X_train, y_train)

ml_accuracy = ml_model.score(X_test, y_test)

# --------------------------------
# DL Model - Neural Network
# --------------------------------

dl_model = Pipeline([
    ("scaler", StandardScaler()),
    (
        "neural_network",
        MLPClassifier(
            hidden_layer_sizes=(10, 5),
            activation="relu",
            max_iter=2000,
            random_state=42
        )
    )
])

dl_model.fit(X_train, y_train)

dl_accuracy = dl_model.score(X_test, y_test)

# --------------------------------
# Prediction History
# --------------------------------

if "history" not in st.session_state:
    st.session_state.history = []

# --------------------------------
# User Input
# --------------------------------

st.header("🩺 Enter Health Details")

col1, col2 = st.columns(2)

with col1:

    age = st.number_input(
        "Age",
        min_value=1,
        max_value=100,
        value=25
    )

    bp = st.number_input(
        "Blood Pressure",
        min_value=50,
        max_value=250,
        value=120
    )

    sugar = st.number_input(
        "Blood Sugar",
        min_value=50,
        max_value=300,
        value=100
    )

with col2:

    cholesterol = st.number_input(
        "Cholesterol",
        min_value=50,
        max_value=400,
        value=180
    )

    bmi = st.number_input(
        "BMI",
        min_value=10.0,
        max_value=60.0,
        value=22.0
    )

# --------------------------------
# Analyze Button
# --------------------------------

if st.button("🔍 Analyze Health", use_container_width=True):

    user_data = pd.DataFrame({
        "age": [age],
        "bp": [bp],
        "sugar": [sugar],
        "cholesterol": [cholesterol],
        "bmi": [bmi]
    })

    # ML Prediction
    ml_result = ml_model.predict(user_data)[0]

    # DL Prediction
    dl_result = dl_model.predict(user_data)[0]

    # --------------------------------
    # AI Decision System
    # --------------------------------

    score = 0

    if bp >= 140:
        score += 1

    if sugar >= 126:
        score += 1

    if cholesterol >= 240:
        score += 1

    if bmi >= 30:
        score += 1

    if score >= 2:
        ai_result = "Higher Risk"
    else:
        ai_result = "Lower Risk"

    # --------------------------------
    # Results
    # --------------------------------

    st.divider()

    st.header("📊 Analysis Results")

    result1, result2, result3 = st.columns(3)

    with result1:

        if ml_result == 1:
            st.error("🤖 ML: Higher Risk")
        else:
            st.success("🤖 ML: Lower Risk")

    with result2:

        if dl_result == 1:
            st.error("🧠 DL: Higher Risk")
        else:
            st.success("🧠 DL: Lower Risk")

    with result3:

        if ai_result == "Higher Risk":
            st.error("💡 AI: Higher Risk")
        else:
            st.success("💡 AI: Lower Risk")

    # --------------------------------
    # Parameter Status
    # --------------------------------

    st.subheader("⚕️ Health Parameter Status")

    p1, p2 = st.columns(2)

    with p1:

        if bp >= 140:
            st.warning("Blood Pressure → ⚠️ High")
        else:
            st.success("Blood Pressure → ✅ Normal")

        if sugar >= 126:
            st.warning("Blood Sugar → ⚠️ High")
        else:
            st.success("Blood Sugar → ✅ Normal")

    with p2:

        if cholesterol >= 240:
            st.warning("Cholesterol → ⚠️ High")
        else:
            st.success("Cholesterol → ✅ Normal")

        if bmi >= 30:
            st.warning("BMI → ⚠️ High")
        else:
            st.success("BMI → ✅ Normal")

    # --------------------------------
    # AI Recommendation
    # --------------------------------

    st.subheader("💡 AI Recommendation")

    if ai_result == "Higher Risk":

        st.info(
            "Some entered health parameters are above the "
            "demo thresholds. Consider discussing unusual "
            "results with a qualified healthcare professional."
        )

    else:

        st.info(
            "The entered values fall within the demo thresholds. "
            "Maintain healthy habits and regular checkups."
        )

    # --------------------------------
    # Prediction History
    # --------------------------------

    risk_status = "High" if (
        ml_result == 1 or dl_result == 1
    ) else "Low"

    st.session_state.history.append({
        "Age": age,
        "BP": bp,
        "Sugar": sugar,
        "Risk": risk_status
    })

    # --------------------------------
    # Chart
    # --------------------------------

    st.subheader("📈 Health Parameters")

    labels = [
        "Blood Pressure",
        "Blood Sugar",
        "Cholesterol",
        "BMI"
    ]

    values = [
        bp,
        sugar,
        cholesterol,
        bmi
    ]

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.bar(labels, values)

    ax.set_ylabel("Values")
    ax.set_title("Health Parameter Analysis")

    plt.xticks(rotation=20)

    st.pyplot(fig)

# --------------------------------
# Prediction History Display
# --------------------------------

if st.session_state.history:

    st.divider()

    st.header("📋 Prediction History")

    history_df = pd.DataFrame(
        st.session_state.history
    )

    st.dataframe(
        history_df,
        use_container_width=True
    )

# --------------------------------
# Model Accuracy
# --------------------------------

st.divider()

st.header("📊 Model Performance")

accuracy_col1, accuracy_col2 = st.columns(2)

with accuracy_col1:

    st.metric(
        "Random Forest ML Accuracy",
        f"{ml_accuracy * 100:.1f}%"
    )

with accuracy_col2:

    st.metric(
        "Neural Network DL Accuracy",
        f"{dl_accuracy * 100:.1f}%"
    )

st.caption(
    "⚠️ This is an educational mini-project and not a medical diagnosis system."
)
