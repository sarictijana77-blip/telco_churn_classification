import os
import pandas as pd
import joblib
import warnings
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
from imblearn.combine import SMOTEENN  # Izmena: SMOTEENN umesto običnog SMOTE za bolji Precision
from imblearn.pipeline import Pipeline as ImbPipeline
import xgboost as xgb

# Ignorisanje upozorenja o promenama verzija
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore', category=FutureWarning)

def find_best_threshold(model, X_val, y_val):
    """
    Funkcija koja koristi validacioni skup da pronađe prag (threshold)
    koji daje najbolji F1-skor, čime balansiramo Precision i Recall.
    """
    # Provera da li model podržava predict_proba (CalibratedSVC i ostali podržavaju)
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_val)[:, 1]
    else:
        probs = model.decision_function(X_val)
        probs = (probs - probs.min()) / (probs.max() - probs.min()) # Skaliranje na 0-1
        
    best_thresh = 0.5
    best_f1 = 0
    
    # Isprobavamo pragove od 0.3 do 0.8
    for thresh in np.arange(0.3, 0.8, 0.02):
        preds = (probs >= thresh).astype(int)
        score = f1_score(y_val, preds)
        if score > best_f1:
            best_f1 = score
            best_thresh = thresh
            
    return best_thresh, best_f1

def train_models(processed_dir, models_dir):
    os.makedirs(models_dir, exist_ok=True)
    
    # Učitavanje trening i validacionog skupa podataka
    train_df = pd.read_csv(os.path.join(processed_dir, 'train.csv'))
    X_train = train_df.drop(columns=['Churn'])
    y_train = train_df['Churn']
    
    val_df = pd.read_csv(os.path.join(processed_dir, 'val.csv'))
    X_val = val_df.drop(columns=['Churn'])
    y_val = val_df['Churn']
    
    # Rečnik u koji ćemo čuvati optimalne pragove za svaki model
    model_thresholds = {}
    
    # =========================================================================
    # 1. LOGISTIC REGRESSION - GridSearchCV
    # =========================================================================
    print("=" * 60)
    print("1/6  Logistic Regression - GridSearchCV (SMOTEENN)")
    print("=" * 60)
    lr_pipeline = ImbPipeline([
        ('smoteenn', SMOTEENN(random_state=42)),
        ('clf', LogisticRegression(random_state=42, max_iter=2000))
    ])
    lr_param_grid = {
        'clf__C': [0.01, 0.1, 1.0, 10.0],
        'clf__solver': ['liblinear', 'lbfgs'],
        'clf__penalty': ['l2'],
    }
    lr_grid = GridSearchCV(lr_pipeline, lr_param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=1)
    lr_grid.fit(X_train, y_train)
    
    # Optimizacija praga na validacionom skupu
    best_lr = lr_grid.best_estimator_
    thresh, val_f1 = find_best_threshold(best_lr, X_val, y_val)
    model_thresholds['logistic_regression_model'] = thresh
    
    print(f"  → Najbolji CV parametri: {lr_grid.best_params_}")
    print(f"  → Optimalni prag na VAL: {thresh:.2f} (F1 na VAL: {val_f1:.4f})")
    joblib.dump(best_lr, os.path.join(models_dir, 'logistic_regression_model.pkl'))
    
    # =========================================================================
    # 2. RANDOM FOREST - GridSearchCV
    # =========================================================================
    print("=" * 60)
    print("2/6  Random Forest - GridSearchCV (SMOTEENN)")
    print("=" * 60)
    rf_pipeline = ImbPipeline([
        ('smoteenn', SMOTEENN(random_state=42)),
        ('clf', RandomForestClassifier(random_state=42))
    ])
    rf_param_grid = {
        'clf__n_estimators': [100, 200],
        'clf__max_depth': [8, 12, 16],
        'clf__min_samples_split': [2, 5],
        'clf__min_samples_leaf': [1, 2],
    }
    rf_grid = GridSearchCV(rf_pipeline, rf_param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=1)
    rf_grid.fit(X_train, y_train)
    
    best_rf = rf_grid.best_estimator_
    thresh, val_f1 = find_best_threshold(best_rf, X_val, y_val)
    model_thresholds['random_forest_model'] = thresh
    
    print(f"  → Najbolji CV parametri: {rf_grid.best_params_}")
    print(f"  → Optimalni prag na VAL: {thresh:.2f} (F1 na VAL: {val_f1:.4f})")
    joblib.dump(best_rf, os.path.join(models_dir, 'random_forest_model.pkl'))
    
    # =========================================================================
    # 3. GRADIENT BOOSTING - GridSearchCV
    # =========================================================================
    print("=" * 60)
    print("3/6  Gradient Boosting - GridSearchCV (SMOTEENN)")
    print("=" * 60)
    gb_pipeline = ImbPipeline([
        ('smoteenn', SMOTEENN(random_state=42)),
        ('clf', GradientBoostingClassifier(random_state=42))
    ])
    gb_param_grid = {
        'clf__n_estimators': [100, 200],
        'clf__learning_rate': [0.05, 0.1],
        'clf__max_depth': [3, 5],
        'clf__subsample': [0.8, 1.0],
    }
    gb_grid = GridSearchCV(gb_pipeline, gb_param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=1)
    gb_grid.fit(X_train, y_train)
    
    best_gb = gb_grid.best_estimator_
    thresh, val_f1 = find_best_threshold(best_gb, X_val, y_val)
    model_thresholds['gradient_boosting_model'] = thresh
    
    print(f"  → Najbolji CV parametri: {gb_grid.best_params_}")
    print(f"  → Optimalni prag na VAL: {thresh:.2f} (F1 na VAL: {val_f1:.4f})")
    joblib.dump(best_gb, os.path.join(models_dir, 'gradient_boosting_model.pkl'))
    
    # =========================================================================
    # 4. KNN - Elbow metoda + GridSearchCV
    # =========================================================================
    print("=" * 60)
    print("4/6  KNN - Elbow + GridSearchCV")
    print("=" * 60)
    print("  → Pokrećem Elbow metodu za odabir optimalnog K na VALIDACIONOM skupu...")
    
    k_values = range(1, 51)
    error_rates_distance = []
    error_rates_uniform = []
    
    for k in k_values:
        knn_dist = KNeighborsClassifier(n_neighbors=k, weights='distance')
        knn_dist.fit(X_train, y_train)
        preds_dist = knn_dist.predict(X_val)  
        error_rates_distance.append(1 - (preds_dist == y_val).mean())
        
        knn_unif = KNeighborsClassifier(n_neighbors=k, weights='uniform')
        knn_unif.fit(X_train, y_train)
        preds_unif = knn_unif.predict(X_val)
        error_rates_uniform.append(1 - (preds_unif == y_val).mean())
    
    best_dist_k = k_values[error_rates_distance.index(min(error_rates_distance))]
    best_unif_k = k_values[error_rates_uniform.index(min(error_rates_uniform))]
    best_dist_error = min(error_rates_distance)
    best_unif_error = min(error_rates_uniform)
    
    if best_dist_error <= best_unif_error:
        optimal_k = best_dist_k
        optimal_weights = 'distance'
        best_error = best_dist_error
    else:
        optimal_k = best_unif_k
        optimal_weights = 'uniform'
        best_error = best_unif_error
    
    print(f"  → Najbolji distance K={best_dist_k} (error: {best_dist_error:.4f})")
    print(f"  → Najbolji uniform K={best_unif_k} (error: {best_unif_error:.4f})")
    print(f"  → Odabrano: K={optimal_k}, weights='{optimal_weights}'")
    
    elbow_df = pd.DataFrame({
        'K': list(k_values),
        'ErrorRate_distance': error_rates_distance,
        'ErrorRate_uniform': error_rates_uniform
    })
    elbow_df.to_csv(os.path.join(models_dir, 'knn_elbow_results.csv'), index=False)
    
    knn_pipeline = ImbPipeline([
        ('smoteenn', SMOTEENN(random_state=42)),
        ('clf', KNeighborsClassifier(weights=optimal_weights))
    ])
    k_range = range(max(1, optimal_k - 4), optimal_k + 5)
    knn_param_grid = {
        'clf__n_neighbors': list(k_range),
        'clf__metric': ['euclidean', 'manhattan', 'minkowski'],
    }
    knn_grid = GridSearchCV(knn_pipeline, knn_param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=1)
    knn_grid.fit(X_train, y_train)
    
    best_knn = knn_grid.best_estimator_
    thresh, val_f1 = find_best_threshold(best_knn, X_val, y_val)
    model_thresholds['knn_model'] = thresh
    
    print(f"  → GridSearch najbolji parametri: {knn_grid.best_params_}")
    print(f"  → Optimalni prag na VAL: {thresh:.2f} (F1 na VAL: {val_f1:.4f})")
    joblib.dump(best_knn, os.path.join(models_dir, 'knn_model.pkl'))
    
    # =========================================================================
    # 5. SVM - GridSearchCV
    # =========================================================================
    print("=" * 60)
    print("5/6  SVM - GridSearchCV (SMOTEENN)")
    print("=" * 60)
    svm_pipeline = ImbPipeline([
        ('smoteenn', SMOTEENN(random_state=42)),
        ('clf', CalibratedClassifierCV(
            estimator=SVC(random_state=42),
            ensemble=False
        ))
    ])
    
    svm_param_grid = {
        'clf__estimator__C': [0.1, 1.0, 10.0],
        'clf__estimator__gamma': ['scale', 'auto'],
        'clf__estimator__kernel': ['rbf', 'poly'],
    }
    svm_grid = GridSearchCV(svm_pipeline, svm_param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=1)
    svm_grid.fit(X_train, y_train)
    
    best_svm = svm_grid.best_estimator_
    thresh, val_f1 = find_best_threshold(best_svm, X_val, y_val)
    model_thresholds['svm_model'] = thresh
    
    print(f"  → Najbolji CV parametri: {svm_grid.best_params_}")
    print(f"  → Optimalni prag na VAL: {thresh:.2f} (F1 na VAL: {val_f1:.4f})")
    joblib.dump(best_svm, os.path.join(models_dir, 'svm_model.pkl'))
    
    # =========================================================================
    # 6. XGBoost - GridSearchCV (UKLONJEN scale_pos_weight ZBOG DUPLIRANJA SA SMOTE)
    # =========================================================================
    print("=" * 60)
    print("6/6  XGBoost - GridSearchCV (Sređeno balansiranje)")
    print("=" * 60)
    
    xgb_pipeline = ImbPipeline([
        ('smoteenn', SMOTEENN(random_state=42)),
        ('clf', xgb.XGBClassifier(random_state=42, eval_metric='logloss'))
    ])
    xgb_param_grid = {
        'clf__n_estimators': [100, 200],
        'clf__learning_rate': [0.05, 0.1, 0.2],
        'clf__max_depth': [3, 6, 9],
        'clf__subsample': [0.8, 1.0],
        'clf__colsample_bytree': [0.8, 1.0],
    }
    xgb_grid = GridSearchCV(xgb_pipeline, xgb_param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=1)
    xgb_grid.fit(X_train, y_train)
    
    best_xgb = xgb_grid.best_estimator_
    thresh, val_f1 = find_best_threshold(best_xgb, X_val, y_val)
    model_thresholds['xgboost_model'] = thresh
    
    print(f"  → Najbolji CV parametri: {xgb_grid.best_params_}")
    print(f"  → Optimalni prag na VAL: {thresh:.2f} (F1 na VAL: {val_f1:.4f})")
    joblib.dump(best_xgb, os.path.join(models_dir, 'xgboost_model.pkl'))
    
    # =========================================================================
    # FAZA: FEATURE SELECTION - Top 5 najvažnijih atributa
    # =========================================================================
    print("=" * 60)
    print("Feature Selection - Top 5 atributa")
    print("=" * 60)
    gb_clf = best_gb.named_steps['clf']
    importances = pd.Series(gb_clf.feature_importances_, index=X_train.columns).sort_values(ascending=False)
    top_features = list(importances.head(5).index)
    
    joblib.dump(top_features, os.path.join(models_dir, 'top_features.pkl'))
    print(f"Odabrani najznačajniji atributi: {top_features}")
    
    print("Treniram redukovani Gradient Boosting model...")
    X_train_reduced = X_train[top_features]
    X_val_reduced = X_val[top_features]
    
    reduced_gb = ImbPipeline([
        ('smoteenn', SMOTEENN(random_state=42)),
        ('clf', GradientBoostingClassifier(
            n_estimators=gb_grid.best_params_['clf__n_estimators'],
            learning_rate=gb_grid.best_params_['clf__learning_rate'],
            max_depth=gb_grid.best_params_['clf__max_depth'],
            subsample=gb_grid.best_params_['clf__subsample'],
            random_state=42
        ))
    ])
    reduced_gb.fit(X_train_reduced, y_train)
    
    thresh, val_f1 = find_best_threshold(reduced_gb, X_val_reduced, y_val)
    model_thresholds['gradient_boosting_reduced_model'] = thresh
    
    print(f"  → Optimalni prag za redukovani model na VAL: {thresh:.2f} (F1 na VAL: {val_f1:.4f})")
    joblib.dump(reduced_gb, os.path.join(models_dir, 'gradient_boosting_reduced_model.pkl'))
    
    # Čuvanje rečnika sa svim pragovima kako bi skripta `evaluate.py` mogla da ih pročita
    joblib.dump(model_thresholds, os.path.join(models_dir, 'optimal_thresholds.pkl'))
    
    print("\n" + "=" * 60)
    print("  SVE FAZE TRENIRANJA I OPTIMIZACIJE PRAGOVA USPEŠNO ZAVRŠENE!")
    print("=" * 60)

if __name__ == "__main__":
    train_models('data/processed', 'models')
