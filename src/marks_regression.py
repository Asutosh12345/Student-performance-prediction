from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd

class MarksRegressionModel:
    """
    Regression model to estimate next exam marks based on student features.
    """
    DEFAULT_FEATURE_COLUMNS = [
        'Attendance',
        'Attendance_Marks',
        'Internal_Marks',
        'Assignment_Marks',
        'Study_Hours',
        'Previous_Results',
        'Previous_Grade_Value',
        'Pass_Fail'
    ]

    def __init__(self):
        self.model = LinearRegression()
        self.trained = False

    def _create_target(self, df: pd.DataFrame):
        if 'Next_Exam_Marks' in df.columns:
            return df['Next_Exam_Marks']
        return df['Internal_Marks'] + df['Assignment_Marks'] + (df['Previous_Results'] * 0.3)

    def train(self, df: pd.DataFrame, feature_columns=None):
        feature_columns = feature_columns or self.DEFAULT_FEATURE_COLUMNS
        missing = [c for c in feature_columns if c not in df.columns]
        if missing:
            raise ValueError(f'Missing feature columns for regression training: {missing}')

        X = df[feature_columns].values
        y = self._create_target(df).values
        self.model.fit(X, y)
        self.trained = True

    def predict(self, df: pd.DataFrame, feature_columns=None):
        if not self.trained:
            raise RuntimeError('Regression model not trained')
        feature_columns = feature_columns or self.DEFAULT_FEATURE_COLUMNS
        missing = [c for c in feature_columns if c not in df.columns]
        if missing:
            raise ValueError(f'Missing feature columns for regression prediction: {missing}')
        X = df[feature_columns].values
        return self.model.predict(X)

    def predict_single(self, features: list):
        if not self.trained:
            raise RuntimeError('Regression model not trained')
        return float(self.model.predict([features])[0])