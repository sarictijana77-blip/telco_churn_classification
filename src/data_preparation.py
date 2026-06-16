import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_and_clean_data(filepath):
    df = pd.read_csv(filepath)
    if 'customerID' in df.columns:
        df = df.drop(columns=['customerID'])
    if 'gender' in df.columns:
        df = df.drop(columns=['gender']) 
    df = df.drop_duplicates()       
        
    # --- REŠENJE ZA GREŠKU (ValueError: could not convert string to float: ' ') ---
    if 'TotalCharges' in df.columns:
        # Zamenjujemo prazne stringove (" ") sa NaN (Not a Number) vrednostima
        df['TotalCharges'] = df['TotalCharges'].replace(r'^\s*$', np.nan, regex=True)
        # Prisilno konvertujemo kolonu u numerički tip (float)
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        # Popunjavamo NaN vrednosti medijanom kolone (robustan način)
        df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
        
    return df

def perform_eda(df, fig_dir):
    """Realizacija faze: Eksplorativna analiza skupa (korelacije i vizuelni prikazi)."""
    os.makedirs(fig_dir, exist_ok=True)
    
    # 1. Vizuelni prikaz: Churn po tipu ugovora (Contract)
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x='Contract', hue='Churn', palette='viridis')
    plt.title('Distribucija Churn-a u zavisnosti od tipa ugovora')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'eda_churn_by_contract.png'))
    plt.close()

    # 2. Korelaciona matrica za numeričke atribute
    plt.figure(figsize=(6, 5))
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    corr_matrix = df[num_cols].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Korelacioni matriks numeričkih atributa')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'eda_correlation_matrix.png'))
    plt.close()
    print("Eksplorativna analiza završena. Grafici su sačuvani u results/figures/")

def preprocess_data(df):
    df = df.copy()
    target_col = 'Churn'
    le_target = LabelEncoder()
    df[target_col] = le_target.fit_transform(df[target_col])
    
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    cat_cols = [col for col in df.columns if col not in num_cols + [target_col]]
    
    # One-Hot Encoding uz izbegavanje zamke dummy promenljivih
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    return df, num_cols, target_col

def split_and_save_data(raw_path, processed_dir, fig_dir):
    df = load_and_clean_data(raw_path)
    
    # Pokretanje EDA pre transformacija
    perform_eda(df, fig_dir)
    
    df_preprocessed, num_cols, target_col = preprocess_data(df)
    
    X = df_preprocessed.drop(columns=[target_col])
    y = df_preprocessed[target_col]
    
    # Podela na Train (70%), Val (15%), Test (15%) sa stratifikacijom
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)
    
    # --- DODATO: .copy() da bi se izbegao "SettingWithCopyWarning" kod skaliranja ---
    X_train = X_train.copy()
    X_val = X_val.copy()
    X_test = X_test.copy()
    
    # Skaliranje (Fit isključivo na train skupu!)
    scaler = StandardScaler()
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_val[num_cols] = scaler.transform(X_val[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])
    
    os.makedirs(processed_dir, exist_ok=True)
    X_train.join(y_train).to_csv(os.path.join(processed_dir, 'train.csv'), index=False)
    X_val.join(y_val).to_csv(os.path.join(processed_dir, 'val.csv'), index=False)
    X_test.join(y_test).to_csv(os.path.join(processed_dir, 'test.csv'), index=False)
    print("Podaci uspešno procesirani i sačuvani!")

if __name__ == "__main__":
    split_and_save_data('data/raw/telco_data.csv', 'data/processed', 'results/figures')