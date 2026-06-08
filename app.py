import os
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Telco Churn Predictor",
    page_icon="📊",
    layout="centered"
)

st.title("Telco Churn Prediction")
st.write("Aplikacija koristi optimizovani Gradient Boosting model za predikciju odlaska korisnika.")
st.write("---")

MODELS_DIR = 'models'
MODEL_PATH = os.path.join(MODELS_DIR, 'gradient_boosting_reduced_model.pkl')
FEATURES_PATH = os.path.join(MODELS_DIR, 'top_features.pkl')

if not os.path.exists(MODEL_PATH) or not os.path.exists(FEATURES_PATH):
    st.error("Model nije pronadjen. Pokrenite prvo train.py.")
else:
    model = joblib.load(MODEL_PATH)
    top_features = joblib.load(FEATURES_PATH)

    st.sidebar.header("Informacije o modelu")
    st.sidebar.success("Model ucitan.")
    st.sidebar.write("Model: Gradient Boosting (Reduced)")
    st.sidebar.write("Broj atributa: 5")

    st.subheader("Unesite podatke o korisniku:")

    user_inputs = {}

    for feature in top_features:
        if feature == 'tenure':
            user_inputs[feature] = st.slider(
                "Koliko meseci je korisnik u mrezi (tenure)?",
                min_value=0, max_value=72, value=12, step=1
            )
        elif feature == 'MonthlyCharges':
            user_inputs[feature] = st.slider(
                "Mesecni troskovi ($):",
                min_value=10.0, max_value=150.0, value=65.0, step=0.5
            )
        elif feature == 'TotalCharges':
            user_inputs[feature] = st.slider(
                "Ukupni troskovi ($):",
                min_value=10.0, max_value=8500.0, value=500.0, step=10.0
            )
        elif feature == 'InternetService_Fiber optic':
            izbor = st.selectbox(
                "Korisnik koristi opticki internet (Fiber Optic)?",
                options=["Ne", "Da"],
                index=0
            )
            user_inputs[feature] = 1 if izbor == "Da" else 0

        elif feature == 'PaymentMethod_Electronic check':
            izbor = st.selectbox(
                "Nacin placanja je elektronski cek (Electronic Check)?",
                options=["Ne", "Da"],
                index=0
            )
            user_inputs[feature] = 1 if izbor == "Da" else 0

        else:
            user_inputs[feature] = st.number_input(
                f"Unesite vrednost za [{feature}]:",
                min_value=0.0, value=0.0
            )

    st.write("")
    if st.button("Pokreni predikciju", use_container_width=True):
        input_df = pd.DataFrame([user_inputs])
        input_df = input_df[top_features]

        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1] * 100

        st.write("---")
        st.subheader("Rezultat:")

        if prediction == 1:
            st.error("Korisnik ce najverovatnije OTI CI (Churn)")
            st.metric(label="Rizik od odlaska", value=f"{probability:.2f}%")
            st.warning("Savet: Ponudite popust ili povoljniji ugovor.")
        else:
            st.success("Korisnik ce najverovatnije OSTATI (No Churn)")
            st.metric(label="Rizik od odlaska", value=f"{probability:.2f}%")
