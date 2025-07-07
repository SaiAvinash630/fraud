import pandas as pd
import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.ensemble import IsolationForest
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, precision_recall_curve, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import shap
import time

from feature_engineering_new import FeatureEngineer

# Features to be engineered by FeatureEngineer
numerical_features = [
    'account_age_days', 'amount', 'quantity', 'total_value',
    'num_trans_24h', 'num_failed_24h', 'new_password_age',
    'freq_last_24h', 'amount_last_24h', 'sudden_category_switch',
    'transaction_hour', 'amount_log', 'amount_to_avg', 'new_device_flag', 'hour_sin', 'hour_cos'
]
categorical_features = ['payment_method', 'device', 'category']

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)

# Full pipeline: feature engineering + preprocessing
full_pipeline = Pipeline([
    ('feature_engineering_new', FeatureEngineer()),
    ('preprocessing', preprocessor)
])

class FraudPredictor:
    def __init__(self, model_path):
        self.model_data = joblib.load(model_path)
        self.xgb_model = self.model_data['xgb_model']
        self.iso_model = self.model_data['iso_model']
        self.pipeline = self.model_data['pipeline']
        self.iso_thresh = self.model_data['iso_thresh']
        self.feature_names = self.model_data['feature_names']
        # SHAP explainer setup
        if hasattr(self.xgb_model, "base_estimator"):
            xgb_for_shap = self.xgb_model.base_estimator
        elif hasattr(self.xgb_model, "estimator"):
            xgb_for_shap = self.xgb_model.estimator
        else:
            xgb_for_shap = self.xgb_model
        self.xgb_explainer = shap.TreeExplainer(xgb_for_shap)

    def predict(self, transaction_data):
        processed = self.pipeline.transform(pd.DataFrame([transaction_data]))
        xgb_prob = self.xgb_model.predict_proba(processed)[0][1]
        iso_score = self.iso_model.decision_function(processed)[0]
        if xgb_prob > 0.9:
            decision = "FRAUD"
        elif (xgb_prob > 0.5 and iso_score < self.iso_thresh):
            decision = "NEED TO TAKE FEEDBACK"
        else:
            decision = "GENUINE"
        shap_values = self.xgb_explainer.shap_values(processed)[0]
        total_impact = np.sum(np.abs(shap_values))
        indicators = []
        for i, val in enumerate(shap_values):
            indicators.append({
                'feature': self.feature_names[i] if i < len(self.feature_names) else f'feature_{i}',
                'value': float(processed[0][i]),
                'impact_percent': round(abs(val) / total_impact * 100, 2) if total_impact > 0 else 0.0
            })
        indicators = sorted(indicators, key=lambda x: x['impact_percent'], reverse=True)[:5]
        fraud_pattern = indicators[0]['feature'] if indicators else "unknown"
        return {
            'decision': decision,
            'probability': round(float(xgb_prob), 4),
            'anomaly_score': round(float(iso_score), 4),
            'fraud_indicators': indicators,
            'fraud_pattern': fraud_pattern,
            'thresholds': {
                'xgb_high': 0.9,
                'xgb_feedback': 0.5,
                'iso_threshold': round(float(self.iso_thresh), 4)
            }
        }

if __name__ == "__main__":
    start_time = time.time()
    # Load data
    df = pd.read_csv('fraud.csv')
    # Drop unnecessary columns
    X = df.drop(['fraud_label', 'Transaction_id','User_id','TimeStamp'], axis=1)
    y = df['fraud_label']
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    # Fit pipeline
    full_pipeline.fit(X_train)
    X_train_processed = full_pipeline.transform(X_train)
    X_test_processed = full_pipeline.transform(X_test)
    # Hyperparameter tuning for XGBoost
    param_grid = {
        'n_estimators': [200, 300, 500],
        'max_depth': [4, 6, 8],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
        'scale_pos_weight': [30, 40, 50]
    }
    xgb = XGBClassifier(random_state=42, eval_metric='auc')
    search = RandomizedSearchCV(xgb, param_grid, n_iter=10, scoring='roc_auc', cv=3, verbose=2, n_jobs=-1)
    search.fit(X_train_processed, y_train)
    xgb_model = search.best_estimator_
    # Cross validation
    cv_scores = cross_val_score(xgb_model, X_train_processed, y_train, cv=5, scoring='roc_auc')
    print(f"Cross-validated ROC-AUC: {cv_scores.mean():.4f}")
    # Calibration
    calibrated_xgb = CalibratedClassifierCV(xgb_model, method='isotonic', cv=3)
    calibrated_xgb.fit(X_train_processed, y_train)
    xgb_model = calibrated_xgb
    # xgb_model.fit(X_train_processed, y_train)
    # Evaluation
    y_pred_proba = xgb_model.predict_proba(X_test_processed)[:, 1]
    y_pred = (y_pred_proba > 0.5).astype(int)
    print("ROC-AUC:", roc_auc_score(y_test, y_pred_proba))
    print("F1-score:", f1_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    # Isolation Forest for anomaly detection
    iso_model = IsolationForest(contamination=0.1, random_state=42)
    iso_model.fit(X_train_processed)
    iso_scores = iso_model.decision_function(X_train_processed)
    iso_thresh = np.percentile(iso_scores, 5)
    # Save models and pipeline
    joblib.dump({
        'xgb_model': xgb_model,
        'iso_model': iso_model,
        'pipeline': full_pipeline,
        'iso_thresh': iso_thresh,
        'feature_names': list(full_pipeline.named_steps['preprocessing'].get_feature_names_out())
    }, 'fraud_detection_model.pkl')
    print(f"Training and saving done in {time.time() - start_time:.1f} seconds.")
    print("Running demo prediction...")
    fraud_predictor = FraudPredictor('fraud_detection_model.pkl')
    sample_transaction = {
        'account_age_days': 30,
        'products': '[{"product_id": "ELE_SMA_001", "category": "Electronics", "product_name": "Smartphone", "base_price": 20000.0, "quantity": 1}]',
        'total_value': 20000.0,
        'payment_method': 'Credit Card',
        'device': 'Laptop',
        'num_trans_24h': 1,
        'num_failed_24h': 0,
        'failed_logins_last_24h': 0,
        'new_password_age': 10,
        'no_of_cards_from_ip': 1,
        'freq_last_24h': 1,
        'amount_last_24h': 20000.0,
        'sudden_category_switch': 0,
        'User_id': 123,
        'TimeStamp': '2024-06-24 14:00:00'
    }
    prediction = fraud_predictor.predict(sample_transaction)
    print("Demo prediction:", prediction)
    print(f"All done! Total time: {time.time() - start_time:.2f} seconds")
