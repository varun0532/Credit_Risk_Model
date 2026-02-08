import pickle
import pandas as pd
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "models", "credit_risk_logreg.pkl")
THRESHOLD_PATH = os.path.join(BASE_DIR, "models", "decision_threshold.pkl")
FEATURE_PATH = os.path.join(BASE_DIR, "models", "model_features.pkl")


class CreditRiskPredictor:
    def __init__(self):
        with open(MODEL_PATH, "rb") as f:
            self.model = pickle.load(f)

        with open(THRESHOLD_PATH, "rb") as f:
            self.threshold = pickle.load(f)

        with open(FEATURE_PATH, "rb") as f:
            self.features = pickle.load(f)

    def preprocess(self, input_dict):
        df = pd.DataFrame([input_dict])

        df = pd.get_dummies(df, drop_first=True)

        # Align input with training features
        df = df.reindex(columns=self.features, fill_value=0)

        return df
    def predict(self, input_dict):

        X = self.preprocess(input_dict)

        # Probability of default
        pd = self.model.predict_proba(X)[0][1]
        non_pd = 1 - pd

        # Credit score calculation (300–900)
        credit_score = int(300 + non_pd * 600)

        # Rating buckets
        if credit_score < 500:
            rating = "Poor"
        elif credit_score < 650:
            rating = "Average"
        elif credit_score < 750:
            rating = "Good"
        else:
            rating = "Excellent"
    
        decision = "High Risk" if pd >= self.threshold else "Low Risk"
    
        return {
        "probability_of_default": round(pd, 3),
        "credit_score": credit_score,
        "rating": rating,
        "risk_label": decision
    }



