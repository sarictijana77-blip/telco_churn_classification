import os
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV

def train_models(processed_dir, models_dir):
    os.makedirs(models_dir, exist_ok=True)
    
    train_df = pd.read_csv(os.path.join(processed_dir, 'train.csv'))
    X_train = train_df.drop(columns=['Churn'])
    y_train = train_df['Churn']
    
    # 1. Baseline pojedinačni model
    print("Treniram Baseline model (Logistic Regression CV)...")
    lr_model = LogisticRegressionCV(cv=5, random_state=42, max_iter=1000)
    lr_model.fit(X_train, y_train)
    joblib.dump(lr_model, os.path.join(models_dir, 'logistic_regression_model.pkl'))
    
    # 2. Ansambl 1: Random Forest sa podešavanjem hiperparametara (GridSearchCV)
    print("Podešavam hiperparametre za Random Forest...")
    rf_param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [6, 10],
        'min_samples_split': [2, 5]
    }
    rf_grid = GridSearchCV(RandomForestClassifier(random_state=42), rf_param_grid, cv=3, scoring='f1', n_jobs=-1)
    rf_grid.fit(X_train, y_train)
    best_rf = rf_grid.best_estimator_
    joblib.dump(best_rf, os.path.join(models_dir, 'random_forest_model.pkl'))
    print(f"Najbolji RF parametri: {rf_grid.best_params_}")
    
    # 3. Ansambl 2: Gradient Boosting sa podešavanjem hiperparametara
    print("Podešavam hiperparametre za Gradient Boosting...")
    gb_param_grid = {
        'n_estimators': [100],
        'learning_rate': [0.05, 0.1],
        'max_depth': [3, 4]
    }
    gb_grid = GridSearchCV(GradientBoostingClassifier(random_state=42), gb_param_grid, cv=3, scoring='f1', n_jobs=-1)
    gb_grid.fit(X_train, y_train)
    best_gb = gb_grid.best_estimator_
    joblib.dump(best_gb, os.path.join(models_dir, 'gradient_boosting_model.pkl'))
    print(f"Najbolji GB parametri: {gb_grid.best_params_}")
    
    # --- FAZA: ODABIR NAJZNAČAJNIJIH ATRIBUTA (Feature Selection) ---
    # Koristimo Gradient Boosting za selekciju top 5 najvažnijih atributa
    importances = pd.Series(best_gb.feature_importances_, index=X_train.columns).sort_values(ascending=False)
    top_features = list(importances.head(5).index)
    
    # Čuvamo spisak top atributa da bi evaluate.py znao koje kolone da izdvoji
    joblib.dump(top_features, os.path.join(models_dir, 'top_features.pkl'))
    print(f"Odabrani najznačajniji atributi: {top_features}")
    
    # Treniramo redukovani model samo na tim atributima radi poređenja (zahtev iz dokumenta)
    print("Treniram redukovani Gradient Boosting model (samo najvažniji atributi)...")
    X_train_reduced = X_train[top_features]
    reduced_gb = GradientBoostingClassifier(**gb_grid.best_params_, random_state=42)
    reduced_gb.fit(X_train_reduced, y_train)
    joblib.dump(reduced_gb, os.path.join(models_dir, 'gradient_boosting_reduced_model.pkl'))
    
    print("Sve faze treniranja i optimizacije uspešno završene!")

if __name__ == "__main__":
    train_models('data/processed', 'models')