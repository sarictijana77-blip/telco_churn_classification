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
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import xgboost as xgb

# Ignorisanje dosadnih upozorenja o promenama verzija
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore', category=FutureWarning)

def train_models(processed_dir, models_dir):
    os.makedirs(models_dir, exist_ok=True)
    
    # Učitavanje trening skupa podataka
    train_df = pd.read_csv(os.path.join(processed_dir, 'train.csv'))
    X_train = train_df.drop(columns=['Churn'])
    y_train = train_df['Churn']
    
    # =========================================================================
    # 1. LOGISTIC REGRESSION - GridSearchCV
    # =========================================================================
    print("=" * 60)
    print("1/6  Logistic Regression - GridSearchCV")
    print("=" * 60)
    lr_pipeline = ImbPipeline([
        ('smote', SMOTE(random_state=42)),
        ('clf', LogisticRegression(class_weight='balanced', random_state=42, max_iter=2000))
    ])
    lr_param_grid = {
        'clf__C': [0.01, 0.1, 1.0, 10.0],
        'clf__solver': ['liblinear', 'lbfgs'],
        'clf__penalty': ['l2'],
    }
    lr_grid = GridSearchCV(lr_pipeline, lr_param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=1)
    lr_grid.fit(X_train, y_train)
    print(f"  → Najbolji parametri: {lr_grid.best_params_}")
    print(f"  → Najbolji F1 (CV): {lr_grid.best_score_:.4f}")
    joblib.dump(lr_grid.best_estimator_, os.path.join(models_dir, 'logistic_regression_model.pkl'))
    
    # =========================================================================
    # 2. RANDOM FOREST - GridSearchCV
    # =========================================================================
    print("=" * 60)
    print("2/6  Random Forest - GridSearchCV")
    print("=" * 60)
    rf_pipeline = ImbPipeline([
        ('smote', SMOTE(random_state=42)),
        ('clf', RandomForestClassifier(random_state=42, class_weight='balanced'))
    ])
    rf_param_grid = {
        'clf__n_estimators': [100, 200],
        'clf__max_depth': [8, 12, 16],
        'clf__min_samples_split': [2, 5],
        'clf__min_samples_leaf': [1, 2],
    }
    rf_grid = GridSearchCV(rf_pipeline, rf_param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=1)
    rf_grid.fit(X_train, y_train)
    print(f"  → Najbolji parametri: {rf_grid.best_params_}")
    print(f"  → Najbolji F1 (CV): {rf_grid.best_score_:.4f}")
    joblib.dump(rf_grid.best_estimator_, os.path.join(models_dir, 'random_forest_model.pkl'))
    
    # =========================================================================
    # 3. GRADIENT BOOSTING - GridSearchCV
    # =========================================================================
    print("=" * 60)
    print("3/6  Gradient Boosting - GridSearchCV")
    print("=" * 60)
    gb_pipeline = ImbPipeline([
        ('smote', SMOTE(random_state=42)),
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
    print(f"  → Najbolji parametri: {gb_grid.best_params_}")
    print(f"  → Najbolji F1 (CV): {gb_grid.best_score_:.4f}")
    joblib.dump(gb_grid.best_estimator_, os.path.join(models_dir, 'gradient_boosting_model.pkl'))
    
    # =========================================================================
    # 4. KNN - Elbow metoda + GridSearchCV (weights, metric)
    # =========================================================================
    print("=" * 60)
    print("4/6  KNN - Elbow + GridSearchCV")
    print("=" * 60)
    print("  → Pokrećem Elbow metodu za odabir optimalnog K na VALIDACIONOM skupu...")
    
    val_df = pd.read_csv(os.path.join(processed_dir, 'val.csv'))
    X_val = val_df.drop(columns=['Churn'])
    y_val = val_df['Churn']
    
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
    print(f"  → Odabrano: K={optimal_k}, weights='{optimal_weights}' (error: {best_error:.4f})")
    
    elbow_df = pd.DataFrame({
        'K': list(k_values),
        'ErrorRate_distance': error_rates_distance,
        'ErrorRate_uniform': error_rates_uniform
    })
    elbow_df.to_csv(os.path.join(models_dir, 'knn_elbow_results.csv'), index=False)
    
    # GridSearch za KNN: dodatno pretraživanje oko optimalnog K + metrika rastojanja
    knn_pipeline = ImbPipeline([
        ('smote', SMOTE(random_state=42)),
        ('clf', KNeighborsClassifier(weights=optimal_weights))
    ])
    k_range = range(max(1, optimal_k - 4), optimal_k + 5)
    knn_param_grid = {
        'clf__n_neighbors': list(k_range),
        'clf__metric': ['euclidean', 'manhattan', 'minkowski'],
    }
    knn_grid = GridSearchCV(knn_pipeline, knn_param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=1)
    knn_grid.fit(X_train, y_train)
    print(f"  → GridSearch najbolji parametri: {knn_grid.best_params_}")
    print(f"  → GridSearch najbolji F1 (CV): {knn_grid.best_score_:.4f}")
    joblib.dump(knn_grid.best_estimator_, os.path.join(models_dir, 'knn_model.pkl'))
    
    # =========================================================================
    # 5. SVM - GridSearchCV (Uklonjen base_estimator zbog sklearn verzije)
    # =========================================================================
    print("=" * 60)
    print("5/6  SVM - GridSearchCV")
    print("=" * 60)
    svm_pipeline = ImbPipeline([
        ('smote', SMOTE(random_state=42)),
        ('clf', CalibratedClassifierCV(
            estimator=SVC(random_state=42, class_weight='balanced'),
            ensemble=False
        ))
    ])
    
    # ISPRAVLJENO: 'clf__base_estimator__...' je zamenjeno sa 'clf__estimator__...'
    svm_param_grid = {
        'clf__estimator__C': [0.1, 1.0, 10.0],
        'clf__estimator__gamma': ['scale', 'auto'],
        'clf__estimator__kernel': ['rbf', 'poly'],
    }
    svm_grid = GridSearchCV(svm_pipeline, svm_param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=1)
    svm_grid.fit(X_train, y_train)
    print(f"  → Najbolji parametri: {svm_grid.best_params_}")
    print(f"  → Najbolji F1 (CV): {svm_grid.best_score_:.4f}")
    joblib.dump(svm_grid.best_estimator_, os.path.join(models_dir, 'svm_model.pkl'))
    
    # =========================================================================
    # 6. XGBoost - GridSearchCV
    # =========================================================================
    print("=" * 60)
    print("6/6  XGBoost - GridSearchCV")
    print("=" * 60)
    scale_pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]
    xgb_pipeline = ImbPipeline([
        ('smote', SMOTE(random_state=42)),
        ('clf', xgb.XGBClassifier(scale_pos_weight=scale_pos_weight, random_state=42, eval_metric='logloss'))
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
    print(f"  → Najbolji parametri: {xgb_grid.best_params_}")
    print(f"  → Najbolji F1 (CV): {xgb_grid.best_score_:.4f}")
    joblib.dump(xgb_grid.best_estimator_, os.path.join(models_dir, 'xgboost_model.pkl'))
    
    # =========================================================================
    # FAZA: FEATURE SELECTION - Top 5 najvažnijih atributa
    # =========================================================================
    print("=" * 60)
    print("Feature Selection - Top 5 atributa")
    print("=" * 60)
    gb_best = gb_grid.best_estimator_
    gb_clf = gb_best.named_steps['clf']
    importances = pd.Series(gb_clf.feature_importances_, index=X_train.columns).sort_values(ascending=False)
    top_features = list(importances.head(5).index)
    
    joblib.dump(top_features, os.path.join(models_dir, 'top_features.pkl'))
    print(f"Odabrani najznačajniji atributi: {top_features}")
    
    # Treniranje redukovanog modela sa najboljim hiperparametrima izvučenim iz GridSearch-a
    print("Treniram redukovani Gradient Boosting model (samo najvažniji atributi)...")
    X_train_reduced = X_train[top_features]
    reduced_gb = ImbPipeline([
        ('smote', SMOTE(random_state=42)),
        ('clf', GradientBoostingClassifier(
            n_estimators=gb_grid.best_params_['clf__n_estimators'],
            learning_rate=gb_grid.best_params_['clf__learning_rate'],
            max_depth=gb_grid.best_params_['clf__max_depth'],
            subsample=gb_grid.best_params_['clf__subsample'],
            random_state=42
        ))
    ])
    reduced_gb.fit(X_train_reduced, y_train)
    joblib.dump(reduced_gb, os.path.join(models_dir, 'gradient_boosting_reduced_model.pkl'))
    
    print("\n" + "=" * 60)
    print("  SVE FAZE TRENIRANJA SA GridSearchCV USPEŠNO ZAVRŠENE!")
    print("=" * 60)

if __name__ == "__main__":
    train_models('data/processed', 'models')