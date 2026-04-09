from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report, 
                             roc_auc_score, roc_curve)
import numpy as np

class StudentPerformanceModel:
    """
    Logistic Regression model for student performance prediction
    """
    
    def __init__(self, random_state=42):
        """
        Initialize the logistic regression model
        
        Args:
            random_state (int): Random seed for reproducibility
        """
        self.model = LogisticRegression(random_state=random_state, max_iter=1000)
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_pred = None
        self.y_pred_proba = None
        
    def train(self, X_train, y_train):
        """
        Train the logistic regression model
        
        Args:
            X_train: Training features
            y_train: Training target variable
        """
        self.X_train = X_train
        self.y_train = y_train
        
        self.model.fit(X_train, y_train)
        print("Model training completed!")
        
    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance on test data
        
        Args:
            X_test: Test features
            y_test: Test target variable
        """
        self.X_test = X_test
        self.y_test = y_test
        
        # Make predictions
        self.y_pred = self.model.predict(X_test)
        self.y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, self.y_pred)
        precision = precision_score(y_test, self.y_pred)
        recall = recall_score(y_test, self.y_pred)
        f1 = f1_score(y_test, self.y_pred)
        roc_auc = roc_auc_score(y_test, self.y_pred_proba)
        
        print("\n" + "="*50)
        print("MODEL EVALUATION METRICS")
        print("="*50)
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        print(f"ROC-AUC:   {roc_auc:.4f}")
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, self.y_pred)
        print(f"\nConfusion Matrix:\n{cm}")
        
        # Classification Report
        print(f"\nClassification Report:\n{classification_report(y_test, self.y_pred, target_names=['Poor Performance', 'Good Performance'])}")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'confusion_matrix': cm
        }
    
    def predict(self, X):
        """
        Make predictions on new data
        
        Args:
            X: Features for prediction
            
        Returns:
            Predictions (0 or 1)
        """
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """
        Predict probability of each class
        
        Args:
            X: Features for prediction
            
        Returns:
            Probability for each class
        """
        return self.model.predict_proba(X)
    
    def get_model_coefficients(self, feature_names=None):
        """
        Display model coefficients and their importance
        
        Args:
            feature_names (list): Names of features
        """
        coefficients = self.model.coef_[0]
        intercept = self.model.intercept_[0]
        
        print("\n" + "="*50)
        print("MODEL COEFFICIENTS (Feature Importance)")
        print("="*50)
        
        if feature_names:
            for name, coef in zip(feature_names, coefficients):
                print(f"{name:20s}: {coef:8.4f}")
        else:
            for i, coef in enumerate(coefficients):
                print(f"Feature {i}: {coef:8.4f}")
        
        print(f"{'Intercept':20s}: {intercept:8.4f}")
        
    def get_roc_curve_data(self):
        """
        Get ROC curve data for visualization
        
        Returns:
            fpr, tpr, thresholds
        """
        if self.y_pred_proba is None:
            print("Model not evaluated yet. Please evaluate first.")
            return None
        
        fpr, tpr, thresholds = roc_curve(self.y_test, self.y_pred_proba)
        return fpr, tpr, thresholds
