import os
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

def evaluate_all_models(processed_dir, models_dir, results_dir):
    fig_dir = os.path.join(results_dir, 'figures')
    metrics_dir = os.path.join(results_dir, 'metrics')
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    
    # 1. Učitavanje testnog skupa podataka
    test_df = pd.read_csv(os.path.join(processed_dir, 'test.csv'))
    X_test = test_df.drop(columns=['Churn'])
    y_test = test_df['Churn']
    
    # Učitavamo spisak top 5 najznačajnijih atributa i optimalne pragove sa validacije
    top_features = joblib.load(os.path.join(models_dir, 'top_features.pkl'))
    
    try:
        thresholds = joblib.load(os.path.join(models_dir, 'optimal_thresholds.pkl'))
        print("[INFO] Uspešno učitani optimalni pragovi sa validacionog skupa.")
    except FileNotFoundError:
        print("[UPOZORENJE] Fajl sa pragovima nije pronađen! Koristiće se podrazumevani prag 0.5.")
        thresholds = {}
    
    # Rečnik modela i pripadajućih testnih podataka
  
    models_to_eval = {
        'Logistic Regression (SMOTEENN)': {
            'model': joblib.load(os.path.join(models_dir, 'logistic_regression_model.pkl')), 
            'data': X_test,
            'thresh_key': 'logistic_regression_model'
        },
        'Random Forest (SMOTEENN)': {
            'model': joblib.load(os.path.join(models_dir, 'random_forest_model.pkl')), 
            'data': X_test,
            'thresh_key': 'random_forest_model'
        },
        'Gradient Boosting (SMOTEENN)': {
            'model': joblib.load(os.path.join(models_dir, 'gradient_boosting_model.pkl')), 
            'data': X_test,
            'thresh_key': 'gradient_boosting_model'
        },
        'KNN (SMOTEENN)': {
            'model': joblib.load(os.path.join(models_dir, 'knn_model.pkl')), 
            'data': X_test,
            'thresh_key': 'knn_model'
        },
        'SVM (SMOTEENN)': {
            'model': joblib.load(os.path.join(models_dir, 'svm_model.pkl')), 
            'data': X_test,
            'thresh_key': 'svm_model'
        },
        'XGBoost (SMOTEENN)': {
            'model': joblib.load(os.path.join(models_dir, 'xgboost_model.pkl')), 
            'data': X_test,
            'thresh_key': 'xgboost_model'
        },
        'Gradient Boosting Reduced (Top 5)': {
            'model': joblib.load(os.path.join(models_dir, 'gradient_boosting_reduced_model.pkl')), 
            'data': X_test[top_features],
            'thresh_key': 'gradient_boosting_reduced_model'
        }
    }
    
    performance_records = []
    
    # 2. Iteracija kroz sve modele i računanje detaljnih metrika uz primenu praga
    for name, config in models_to_eval.items():
        model = config['model']
        X_data = config['data']
        thresh_key = config['thresh_key']
        
        # Dobijanje optimalnog praga za konkretan model (ako ne postoji, default je 0.5)
        current_thresh = thresholds.get(thresh_key, 0.5)
        
        # Računanje verovatnoća (ili decision funkcije za specifične slučajeve)
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_data)[:, 1]
        else:
            y_prob = model.decision_function(X_data)
            y_prob = (y_prob - y_prob.min()) / (y_prob.max() - y_prob.min())
        
        # PRIMENA OPTIMALNOG PRAGA UMESTO STREKTNOG model.predict()
        y_pred = (y_prob >= current_thresh).astype(int)
        
        # Izračunavanje svih metrika
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)
        
        metrics = {
            'Model': name,
            'Threshold': round(current_thresh, 2),
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1-Score': f1,
            'ROC-AUC': roc_auc
        }
        performance_records.append(metrics)
        
        # 3. Kreiranje i vizuelno doterivanje Matrice Konfuzije
        plt.figure(figsize=(6, 5))
        cm = confusion_matrix(y_test, y_pred)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', 
                    xticklabels=['No Churn', 'Churn'], 
                    yticklabels=['No Churn', 'Churn'], cbar=False)
        
        plt.title(f'Matrica konfuzije: {name} (Prag: {current_thresh:.2f})\n'
                  f'Acc: {acc:.3f} | Prec: {prec:.3f} | Rec: {rec:.3f} | F1: {f1:.3f}', 
                  fontsize=9, pad=10)
        plt.ylabel('Stvarna klasa (Ground Truth)')
        plt.xlabel('Predviđena klasa (Model Prediction)')
        plt.tight_layout()
        
        filename = name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("-", "").replace("+", "plus")
        plt.savefig(os.path.join(fig_dir, f'{filename}_confusion_matrix.png'), dpi=300)
        plt.close()

    # 4. Generisanje i čuvanje finalnog CSV izveštaja
    report_df = pd.DataFrame(performance_records)
    report_df.to_csv(os.path.join(metrics_dir, 'final_model_comparison_report.csv'), index=False)
    
    print("\n" + "="*85)
    print(" ZVANIČNI REZULTATI EVALUACIJE ZA PROJEKAT (SAUSAU 2026) - KORIGOVANI PRAGOVI")
    print("="*85)
    print(report_df.to_string(index=False))
    print("="*85)
    print(f"Sve matrice konfuzije sa ispisanim metrikama su sačuvane u: {fig_dir}")
    print(f"Uporedni CSV izveštaj je sačuvan u: {metrics_dir}\n")

if __name__ == "__main__":
    evaluate_all_models('data/processed', 'models', 'results')
