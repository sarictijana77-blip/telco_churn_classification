import os
import pandas as pd
import numpy as np
import streamlit as st
import joblib
from sklearn.preprocessing import StandardScaler

st.set_page_config(
    page_title="Telco Churn Predictor",
    page_icon="📊",
    layout="centered"
)

st.title("Telco Churn Prediction")
st.write("Aplikacija koristi optimizovani Gradient Boosting model sa optimalnim pragom i ranom uzbunom za rizične korisnike.")
st.write("---")

MODELS_DIR = 'models'
MODEL_PATH = os.path.join(MODELS_DIR, 'gradient_boosting_reduced_model.pkl')
FEATURES_PATH = os.path.join(MODELS_DIR, 'top_features.pkl')
RAW_DATA_PATH = 'data/raw/telco_data.csv'

# Provera da li svi neophodni fajlovi postoje na disku
if not os.path.exists(MODEL_PATH) or not os.path.exists(FEATURES_PATH):
    st.error("Model nije pronađen. Pokrenite prvo train.py.")
elif not os.path.exists(RAW_DATA_PATH):
    st.error("Sirovi podaci (data/raw/telco_data.csv) nisu pronađeni. Neophodni su za ispravno skaliranje u Streamlit-u!")
else:
    # Učitavanje modela i top 5 atributa
    model = joblib.load(MODEL_PATH)
    top_features = joblib.load(FEATURES_PATH)

    # Učitavanje optimalnog praga iz datoteke optimal_thresholds.pkl
    OPTIMAL_THRESHOLD_PATH = os.path.join(MODELS_DIR, 'optimal_thresholds.pkl')
    if os.path.exists(OPTIMAL_THRESHOLD_PATH):
        thresholds = joblib.load(OPTIMAL_THRESHOLD_PATH)
        optimal_threshold = thresholds.get('gradient_boosting_reduced_model', 0.58)
    else:
        # Podrazumevani prag prema tvom modelu je 0.58
        optimal_threshold = 0.58

    # Prikaz informacija u sidebar-u
    st.sidebar.header("Informacije o modelu")
    st.sidebar.success("Model uspešno učitan.")
    st.sidebar.write("**Model:** Gradient Boosting (Reduced)")
    st.sidebar.write(f"**Broj atributa:** {len(top_features)}")
    st.sidebar.write(f"**Optimalni prag (VAL):** {optimal_threshold:.2f}")
    st.sidebar.markdown("""
    ---
    *Sistem rane uzbune:* Korisnici sa rizikom između 40% i 57% biće označeni kao potencijalno nestabilni (siva zona).
    """)

    st.subheader("Unesite podatke o korisniku:")

    user_inputs = {}

    # Dinamičko kreiranje interfejsa na osnovu top_features koje je selektovao train.py
    for feature in top_features:
        if feature == 'tenure':
            user_inputs[feature] = st.slider(
                "Koliko meseci je korisnik u mreži (tenure)?",
                min_value=0, max_value=72, value=12, step=1
            )
        elif feature == 'MonthlyCharges':
            user_inputs[feature] = st.slider(
                "Mesečni troškovi ($):",
                min_value=10.0, max_value=150.0, value=65.0, step=0.5
            )
        elif feature == 'TotalCharges':
            user_inputs[feature] = st.slider(
                "Ukupni troškovi ($):",
                min_value=10.0, max_value=8500.0, value=500.0, step=10.0
            )
        elif feature == 'InternetService_Fiber optic':
            izbor = st.selectbox(
                "Korisnik koristi optički internet (Fiber Optic)?",
                options=["Ne", "Da"],
                index=0
            )
            user_inputs[feature] = 1 if izbor == "Da" else 0

        elif feature == 'Contract_Two year':
            izbor = st.selectbox(
                "Korisnik ima ugovor na 2 godine (Contract Two Year)?",
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
        # 1. Kreiranje DataFrame-a od korisničkih unosa
        input_df = pd.DataFrame([user_inputs])
        input_df = input_df[top_features]

        # 2. SKALIRANJE PODATAKA U REALNOM VREMENU (Usklađivanje sa obradom iz data_preparation.py)
        try:
            raw_df = pd.read_csv(RAW_DATA_PATH)
            raw_df['TotalCharges'] = raw_df['TotalCharges'].replace(r'^\s*$', np.nan, regex=True)
            raw_df['TotalCharges'] = pd.to_numeric(raw_df['TotalCharges'], errors='coerce')
            raw_df['TotalCharges'] = raw_df['TotalCharges'].fillna(raw_df['TotalCharges'].median())
            
            base_scaler = StandardScaler()
            base_scaler.fit(raw_df[['tenure', 'MonthlyCharges', 'TotalCharges']])
            
            all_num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
            for col in top_features:
                if col in all_num_cols:
                    idx = all_num_cols.index(col)
                    mean_val = base_scaler.mean_[idx]
                    std_val = base_scaler.scale_[idx]
                    
                    # Standardizacija: (x - mean) / std
                    input_df[col] = (input_df[col] - mean_val) / std_val
                    
        except Exception as e:
            st.error(f"Greška prilikom skaliranja podataka: {e}")

        # 3. Izvršavanje predikcije i klasifikacija pomoću OPTIMALNOG PRAGA
        prob_churn = model.predict_proba(input_df)[0][1]
        prediction = 1 if prob_churn >= optimal_threshold else 0
        prob_pct = prob_churn * 100

        st.write("---")
        st.subheader("Rezultat:")

        # Zvanična klasifikacija na osnovu praga od 58%
        if prediction == 1:
            st.error("Korisnik će najverovatnije OTIĆI (Churn)")
            st.metric(label="Izračunati rizik od odlaska", value=f"{prob_pct:.1f}%", delta=f"+{(prob_pct - (optimal_threshold*100)):.1f}% iznad praga")
            st.warning("Savet: Korisnik je prešao kritičnu granicu. Hitno ponudite direktan popust, prelazak na dvogodišnji ugovor ili targetiranu restenzionu ponudu.")
        else:
            st.success("Korisnik će najverovatnije OSTATI (No Churn)")
            st.metric(label="Izračunati rizik od odlaska", value=f"{prob_pct:.1f}%", delta=f"{(prob_pct - (optimal_threshold*100)):.1f}% ispod praga")
            
            # ISPRAVLJENA LOGIKA: Rana uzbuna pokriva tačan opseg od 40.0% do 57.0%
            if 40.0 <= prob_pct <= 57.0:
                st.info("⚠️ **RANA UZBUNA (Siva zona rizika: 40% - 57%):**")
                st.write("**Savet za kompaniju:** Iako je korisnik tehnički i dalje ispod zvaničnog praga odlaska od 58%, njegov izračunati rizik je visok i nalazi se u sivoj zoni. Preporučuje se preventivno delovanje: kontaktirajte korisnika, pošaljite automatizovanu anketu o zadovoljstvu ili ponudite manju pogodnost (npr. besplatne gigabajte ili popust na naredni račun) kako biste sprečili da rizik pređe kritičnu granicu.")