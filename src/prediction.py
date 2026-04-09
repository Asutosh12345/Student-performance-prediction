import pandas as pd
import numpy as np

class PredictionEngine:
    """
    Engine for making predictions on new student data
    """
    
    def __init__(self, model, scaler, feature_names=None):
        """
        Initialize the prediction engine
        
        Args:
            model: Trained logistic regression model
            scaler: StandardScaler fitted on training data
            feature_names (list): Names of features
        """
        self.model = model
        self.scaler = scaler
        self.feature_names = feature_names or [
            'Attendance', 'Internal_Marks', 'Assignment_Marks', 'Study_Hours', 'Previous_Results'
        ]
    
    def predict_single_student(self, attendance, internal_marks, assignment_marks, 
                               study_hours, previous_results):
        """
        Predict performance for a single student
        
        Args:
            attendance (float): Attendance percentage (0-100)
            internal_marks (float): Internal marks (0-50)
            assignment_marks (float): Assignment marks (0-20)
            study_hours (float): Study hours per week (0-10)
            previous_results (float): Previous results (0-100)
            
        Returns:
            dict: Prediction result with probability
        """
        # Create a DataFrame for a single student
        student_data = pd.DataFrame({
            'Attendance': [attendance],
            'Internal_Marks': [internal_marks],
            'Assignment_Marks': [assignment_marks],
            'Study_Hours': [study_hours],
            'Previous_Results': [previous_results]
        })
        
        # Scale the data
        student_scaled = self.scaler.transform(student_data)
        
        # Make prediction
        prediction = self.model.predict(student_scaled)[0]
        probability = self.model.predict_proba(student_scaled)[0]
        
        result = {
            'Performance': 'Good' if prediction == 1 else 'Poor',
            'Confidence': max(probability) * 100,
            'Poor_Performance_Probability': probability[0] * 100,
            'Good_Performance_Probability': probability[1] * 100
        }
        
        return result
    
    def predict_batch(self, data_df):
        """
        Predict performance for multiple students
        
        Args:
            data_df (pd.DataFrame): DataFrame with student data
            
        Returns:
            pd.DataFrame: DataFrame with predictions
        """
        # Scale the data
        data_scaled = self.scaler.transform(data_df)
        
        # Make predictions
        predictions = self.model.predict(data_scaled)
        probabilities = self.model.predict_proba(data_scaled)
        
        # Create result DataFrame
        result_df = data_df.copy()
        result_df['Predicted_Performance'] = ['Good' if p == 1 else 'Poor' for p in predictions]
        result_df['Confidence'] = np.max(probabilities, axis=1) * 100
        result_df['Good_Performance_Probability'] = probabilities[:, 1] * 100
        
        return result_df
    
    def identify_weak_students(self, data_df, threshold=0.4):
        """
        Identify students who may need improvement
        
        Args:
            data_df (pd.DataFrame): DataFrame with student data
            threshold (float): Probability threshold for identifying weak students
            
        Returns:
            pd.DataFrame: DataFrame of students needing improvement
        """
        predictions_df = self.predict_batch(data_df)
        
        # Filter students with probability of good performance below threshold
        weak_students = predictions_df[
            predictions_df['Good_Performance_Probability'] < (threshold * 100)
        ]
        
        return weak_students.sort_values('Good_Performance_Probability')
    
    def generate_report(self, data_df):
        """
        Generate a comprehensive performance report
        
        Args:
            data_df (pd.DataFrame): DataFrame with student data
            
        Returns:
            dict: Report with statistics
        """
        predictions_df = self.predict_batch(data_df)
        
        good_students = len(predictions_df[predictions_df['Predicted_Performance'] == 'Good'])
        poor_students = len(predictions_df[predictions_df['Predicted_Performance'] == 'Poor'])
        
        report = {
            'total_students': len(predictions_df),
            'good_performance': good_students,
            'poor_performance': poor_students,
            'good_performance_percentage': (good_students / len(predictions_df)) * 100,
            'poor_performance_percentage': (poor_students / len(predictions_df)) * 100,
            'average_confidence': predictions_df['Confidence'].mean(),
            'predictions_df': predictions_df
        }
        
        return report
    
    def print_prediction_result(self, result):
        """
        Pretty print prediction result
        
        Args:
            result (dict): Prediction result from predict_single_student
        """
        print("\n" + "="*50)
        print("PREDICTION RESULT")
        print("="*50)
        print(f"Performance: {result['Performance']}")
        print(f"Confidence: {result['Confidence']:.2f}%")
        print(f"Good Performance Probability: {result['Good_Performance_Probability']:.2f}%")
        print(f"Poor Performance Probability: {result['Poor_Performance_Probability']:.2f}%")
        print("="*50 + "\n")
