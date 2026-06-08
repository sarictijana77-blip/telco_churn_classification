import os
import pandas as pd
import joblib
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
    
    # Učitavamo spisak top 5 najznačajnijih atributa koji je napravio train.py
    top_features = joblib.load(os.path.join(models_dir, 'top_features.pkl'))
    
    # Rečnik modela i pripadajućih testnih podataka (Full vs Reduced)
    models_to_eval = {
        'Logistic Regression (Baseline)': (joblib.load(os.path.join(models_dir, 'logistic_regression_model.pkl')), X_test),
        'Random Forest (Full)': (joblib.load(os.path.join(models_dir, 'random_forest_model.pkl')), X_test),
        'Gradient Boosting (Full)': (joblib.load(os.path.join(models_dir, 'gradient_boosting_model.pkl')), X_test),
        'Gradient Boosting (Reduced - Top 5)': (joblib.load(os.path.join(models_dir, 'gradient_boosting_reduced_model.pkl')), X_test[top_features])
    }
    
    performance_records = []
    
    # 2. Iteracija kroz sve modele i računanje detaljnih metrika
    for name, (model, X_data) in models_to_eval.items():
        y_pred = model.predict(X_data)
        y_prob = model.predict_proba(X_data)[:, 1]
        
        # Izračunavanje svih traženih metrika za klasifikaciju
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)
        
        metrics = {
            'Model': name,
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
        
        # Heatmap vizuelizacija matrice
        sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', 
                    xticklabels=['No Churn', 'Churn'], 
                    yticklabels=['No Churn', 'Churn'], cbar=False)
        
        # Dodavanje izračunatih metrika direktno ispod naslova grafika radi bolje dokumentacije
        plt.title(f'Matrica konfuzije: {name}\n'
                  f'Acc: {acc:.3f} | Prec: {prec:.3f} | Rec: {rec:.3f} | F1: {f1:.3f}', 
                  fontsize=10, pad=10)
        plt.ylabel('Stvarna klasa (Ground Truth)')
        plt.xlabel('Predviđena klasa (Model Prediction)')
        plt.tight_layout()
        
        # Bezbedno generisanje imena fajla na osnovu imena modela
        filename = name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("-", "")
        plt.savefig(os.path.join(fig_dir, f'{filename}_confusion_matrix.png'), dpi=300)
        plt.close()

    # 4. Generisanje i čuvanje finalnog CSV izveštaja sa svim metrikama
    report_df = pd.DataFrame(performance_records)
    report_df.to_csv(os.path.join(metrics_dir, 'final_model_comparison_report.csv'), index=False)
    
    print("\n" + "="*70)
    print(" ZVANIČNI REZULTATI EVALUACIJE ZA PROJEKAT (SAUSAU 2026)")
    print("="*70)
    print(report_df.to_string(index=False))
    print("="*70)
    print(f"Sve matrice konfuzije sa ispisanim metrikama su sačuvane u: {fig_dir}")
    print(f"Uporedni CSV izveštaj je sačuvan u: {metrics_dir}\n")

if __name__ == "__main__":
    evaluate_all_models('data/processed', 'models', 'results')