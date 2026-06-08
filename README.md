# Telco Customer Churn Classification

Projektni zadatak - predikcija odlaska korisnika (churn) telekomunikacione mreze koriscenjem ML modela.

## Struktura projekta
```
c:\SAUSAU\
├── data/
│   ├── raw/                    Ulazni podaci (telco_data.csv)
│   └── processed/              Procesirani podaci (train.csv, val.csv, test.csv)
├── models/                     Trenirani modeli (.pkl)
├── results/
│   ├── figures/                Grafici (EDA, matrice konfuzije)
│   └── metrics/                CSV izvestaji sa metrikama
├── src/
│   ├── data_preparation.py     Priprema i preprocessing podataka
│   ├── train.py                Treniranje modela
│   └── evaluate.py             Evaluacija modela
├── app.py                      Web aplikacija (Streamlit)
├── requirements.txt            Python zavisnosti
└── README.md                   Ovo uputstvo
```

## Instalacija

```bash
pip install -r requirements.txt
```

## Pipeline pokretanja

Svaki skript se pokrece redom, jedan za drugim.

### 1. data_preparation.py

```bash
python src/data_preparation.py
```

Skript ucitava fajl `data/raw/telco_data.csv`, cisti podatke (uklanja customerID, konvertuje TotalCharges u broj, popunjava missing vrednosti), radi EDA (graf distribucije churn-a po tipu ugovora i korelaciona matrica), radi One-Hot Encoding za kategoricke atribute, deli podatke na train/val/test (70/15/15) sa stratifikacijom, skalira numericke vrednosti pomocu StandardScaler-a, i cuva rezultat u `data/processed/`.

### 2. train.py

```bash
python src/train.py
```

Skript ucitava `data/processed/train.csv` i trenira tri modela:
- Logistic Regression (baseline, sa unakrsnom validacijom)
- Random Forest (GridSearchCV za hiperparametre: n_estimators, max_depth, min_samples_split)
- Gradient Boosting (GridSearchCV za hiperparametre: learning_rate, max_depth)

Zatim koristi Gradient Boosting da odredi top 5 najvaznijih atributa i trenira redukovani Gradient Boosting model samo na tim atributima. Svi modeli se cuvaju u `models/` folderu.

### 3. evaluate.py

```bash
python src/evaluate.py
```

Skript ucitava `data/processed/test.csv` i sve modele iz `models/` foldera. Za svaki model racuna: Accuracy, Precision, Recall, F1-Score i ROC-AUC. Generise matrice konfuzije (sacuvane u `results/figures/`) i uporedni CSV izvestaj (`results/metrics/final_model_comparison_report.csv`).

### 4. app.py

```bash
streamlit run app.py
```

Pokrece web aplikaciju koja ucitava redukovani Gradient Boosting model (sa top 5 atributa). Korisnik unosi podatke o korisniku kroz formu (tenure, MonthlyCharges, TotalCharges, InternetService, PaymentMethod), a aplikacija prikazuje da li ce korisnik otici ili ostati, zajedno sa procentom rizika.

## Brzi start

```bash
pip install -r requirements.txt
python src/data_preparation.py
python src/train.py
python src/evaluate.py
streamlit run app.py
```

## Tehnicki detalji

- Python 3.14+
- scikit-learn za ML modele
- Streamlit za web aplikaciju
- StandardScaler za normalizaciju
- Stratified train/val/test split (70/15/15)
