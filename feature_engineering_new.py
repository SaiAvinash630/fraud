from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd
import json

def get_amount(s):
    try:
        products = json.loads(s)
        return sum(p.get('base_price', 0) * p.get('quantity', 1) for p in products)
    except Exception:
        return 0

def get_quantity(s):
    try:
        products = json.loads(s)
        return sum(p.get('quantity', 1) for p in products)
    except Exception:
        return 0

def get_category(s):
    try:
        products = json.loads(s)
        cats = sorted(set(p['category'] for p in products if 'category' in p))
        return '|'.join(cats) if cats else 'Unknown'
    except Exception:
        return 'Unknown'

class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        # Store user average amount if needed
        if 'User_id' in X.columns and 'amount' in X.columns:
            self.user_mean_amount = X.groupby('User_id')['amount'].mean()
        else:
            self.user_mean_amount = None
        return self

    def transform(self, X):
        X = X.copy()

        # Extract amount and quantity from products
        if 'products' in X.columns:
            X['amount'] = X['products'].apply(get_amount)
            X['quantity'] = X['products'].apply(get_quantity)
            X['category'] = X['products'].apply(get_category)
        else:
            X['amount'] = 0
            X['quantity'] = 0
            X['category'] = 'Unknown'

        # Transaction hour
        if 'TimeStamp' in X.columns:
            X['transaction_hour'] = pd.to_datetime(X['TimeStamp']).dt.hour
        else:
            X['transaction_hour'] = 0

        # Log amount
        X['amount_log'] = np.log1p(X['amount'])

        # Amount to user average amount ratio
        if self.user_mean_amount is not None and 'User_id' in X.columns:
            X['amount_to_avg'] = X.apply(
                lambda row: row['amount'] / self.user_mean_amount.get(row['User_id'], row['amount']),
                axis=1
            )
        else:
            X['amount_to_avg'] = 1.0

        # New device flag: if device changed for user compared to previous transaction
        if 'User_id' in X.columns and 'device' in X.columns and 'TimeStamp' in X.columns:
            X = X.sort_values(['User_id', 'TimeStamp'])
            X['new_device_flag'] = (
            X.groupby('User_id')['device'].transform(lambda s: s != s.shift(1)).astype(int).fillna(0))
        else:
            X['new_device_flag'] = 0

        # Hour cyclical features
        X['hour_sin'] = np.sin(2 * np.pi * X['transaction_hour'] / 24)
        X['hour_cos'] = np.cos(2 * np.pi * X['transaction_hour'] / 24)

        # Fill missing numeric columns with 0 (based on your dataset fields)
        numeric_cols = [
            'account_age_days', 'amount', 'quantity', 'total_value', 'promo_used', 
            'num_trans_24h', 'num_failed_24h', 'failed_logins_last_24h', 'new_password_age',
            'freq_last_24h', 'amount_last_24h', 'sudden_category_switch', 'transaction_hour',
            'amount_log', 'amount_to_avg', 'new_device_flag', 'hour_sin', 'hour_cos'
        ]

        for col in numeric_cols:
            if col not in X.columns:
                X[col] = 0
            else:
                # convert to numeric if not already
                X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)

        # Fill missing categorical columns with 'Unknown'
        categorical_cols = ['payment_method', 'device', 'category']
        for col in categorical_cols:
            if col not in X.columns:
                X[col] = 'Unknown'
            else:
                X[col] = X[col].fillna('Unknown').astype(str)

        return X
