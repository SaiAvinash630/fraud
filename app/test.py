import joblib
import shap
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd

from feature_engineering import FeatureEngineer  # Adjust the import based on your project structure
# 1. Define the FraudPredictor class
class FraudPredictor:
    def __init__(self, model_path):
        self.model_data = joblib.load(model_path)
        self.xgb_model = self.model_data['xgb_model']
        self.iso_model = self.model_data['iso_model']
        self.pipeline = self.model_data['pipeline']
        self.iso_thresh = self.model_data['iso_thresh']
        self.feature_names = self.model_data['feature_names']
        # Use the underlying estimator for SHAP
        if hasattr(self.xgb_model, "base_estimator_"):
            shap_model = self.xgb_model.base_estimator_
        elif hasattr(self.xgb_model, "estimator"):
            shap_model = self.xgb_model.estimator
        else:
            shap_model = self.xgb_model
        self.xgb_explainer = shap.TreeExplainer(shap_model)
        
    def predict(self, transaction_data):
        processed = self.pipeline.transform(pd.DataFrame([transaction_data]))
        xgb_prob = self.xgb_model.predict_proba(processed)[0][1]
        iso_score = self.iso_model.decision_function(processed)[0]
        
        # Decision logic
        if xgb_prob > 0.8:
            decision = "FRAUD"
        elif (xgb_prob > 0.6 and iso_score < self.iso_thresh):
            decision = "NEED TO TAKE FEEDBACK"
        else:
            decision = "GENUINE"
        
        # SHAP explanation
        shap_values = self.xgb_explainer.shap_values(processed)[0]
        indicators = []
        total_impact = sum(np.abs(shap_values))
        for i, val in enumerate(shap_values):
            indicators.append({
                'feature': self.feature_names[i] if i < len(self.feature_names) else f'feature_{i}',
                'value': float(processed[0][i]),
                'impact_percent': round((abs(val)/total_impact)*100, 2) if total_impact else 0.0
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
                'xgb_high': 0.8,
                'xgb_feedback': 0.6,
                'iso_threshold': round(float(self.iso_thresh), 4)
            }
        }

# 2. Load the model
fraud_predictor = FraudPredictor('models/hybrid.pkl')

# 3. Prepare your transaction (raw parameters)
sample_transaction = {
    'account_age_days': 1,
    'payment_method': 'Credit Card',
    'device': 'Laptop',
    'category': 'Electronics',
    'amount': 1000.0,
    'quantity': 3,
    'total_value': 3000.0,
    'num_trans_24h': 1,
    'num_failed_24h': 0,
    'no_of_cards_from_ip': 1,
    'promo_used': 0,
    'freq_last_24h': 1,
    'amount_last_24h': 500,
    'sudden_category_switch': 0,
    'User_id': 123,
    'TimeStamp': '2024-06-24 14:00:00'
}

# 4. Run prediction
prediction = fraud_predictor.predict(sample_transaction)
print(prediction)