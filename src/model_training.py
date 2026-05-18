from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report,
                             roc_auc_score)
import numpy as np


class StudentPerformanceModel:
    """
    Logistic Regression model for student performance prediction.
    """

    def __init__(self, random_state=42):
        self.model = LogisticRegression(
            random_state=random_state,
            max_iter=2000,
            solver='lbfgs',
            class_weight='balanced',   # FIX 7: handle class imbalance
        )
        self.X_train = None
        self.X_test  = None
        self.y_train = None
        self.y_test  = None
        self.y_pred  = None
        self.y_pred_proba = None

    def train(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.model.fit(X_train, y_train)
        print("Model training completed!")

    def evaluate(self, X_test, y_test):
        self.X_test = X_test
        self.y_test = y_test

        self.y_pred       = self.model.predict(X_test)
        self.y_pred_proba = self.model.predict_proba(X_test)

        accuracy  = accuracy_score(y_test, self.y_pred)
        precision = precision_score(y_test, self.y_pred, average='weighted', zero_division=0)
        recall    = recall_score   (y_test, self.y_pred, average='weighted', zero_division=0)
        f1        = f1_score       (y_test, self.y_pred, average='weighted', zero_division=0)

        # FIX 8: roc_auc_score for multiclass always needs multi_class='ovr' and
        # the full probability matrix.  The old code passed the full proba matrix
        # to the binary-only path when n_classes > 2, raising a ValueError.
        n_classes = self.y_pred_proba.shape[1]
        if n_classes == 2:
            roc_auc = roc_auc_score(y_test, self.y_pred_proba[:, 1])
        else:
            roc_auc = roc_auc_score(
                y_test, self.y_pred_proba,
                multi_class='ovr', average='weighted'
            )

        print("\n" + "=" * 50)
        print("MODEL EVALUATION METRICS")
        print("=" * 50)
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        print(f"ROC-AUC:   {roc_auc:.4f}")

        cm = confusion_matrix(y_test, self.y_pred)
        print(f"\nConfusion Matrix:\n{cm}")

        unique_labels = sorted(list(set(y_test)))
        label_names   = []
        for label in unique_labels:
            if   label == 0: label_names.append('Poor Performance')
            elif label == 1: label_names.append('Average Performance')
            elif label == 2: label_names.append('Good Performance')
            else:            label_names.append(f'Class {label}')

        print(f"\nClassification Report:\n"
              f"{classification_report(y_test, self.y_pred, labels=unique_labels, target_names=label_names)}")

        return {
            'accuracy':         accuracy,
            'precision':        precision,
            'recall':           recall,
            'f1_score':         f1,
            'roc_auc':          roc_auc,
            'confusion_matrix': cm,
        }

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def get_model_coefficients(self, feature_names=None):
        # For multinomial LR, coef_ has shape (n_classes, n_features).
        # Show average absolute importance across classes.
        mean_abs_coef = np.mean(np.abs(self.model.coef_), axis=0)
        print("\n" + "=" * 50)
        print("MODEL COEFFICIENTS (mean |coef| across classes)")
        print("=" * 50)
        if feature_names:
            for name, coef in zip(feature_names, mean_abs_coef):
                print(f"{name:25s}: {coef:8.4f}")
        else:
            for i, coef in enumerate(mean_abs_coef):
                print(f"Feature {i}: {coef:8.4f}")
        print(f"{'Intercept':25s}: {np.mean(self.model.intercept_):8.4f}")