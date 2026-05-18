import pandas as pd
import numpy as np

class PredictionEngine:
    """
    Engine for making predictions on new student data
    """

    DEFAULT_FEATURE_NAMES = [
        'Attendance',
        'Attendance_Marks',
        'Internal_Marks',
        'Previous_Results',
        'Previous_Grade_Value'
    ]

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
        self.feature_names = feature_names or self.DEFAULT_FEATURE_NAMES
        self.regression_model = None

    def _ensure_feature_frame(self, input_df):
        """Normalize DataFrame columns before prediction."""
        df = input_df.copy()
        for col in self.feature_names:
            if col not in df.columns:
                df[col] = 0.0
        df = df.reindex(columns=self.feature_names, fill_value=0.0)
        return df.astype(float)

    def _get_category_from_marks(self, marks):
        if marks < 50:
            return 'Poor'
        if marks < 75:
            return 'Average'
        return 'Good'

    def _get_category_from_probability(self, good_probability):
        if good_probability >= 0.75:
            return 'Good'
        if good_probability >= 0.40:
            return 'Average'
        return 'Poor'

    def _get_category_from_label(self, label):
        return {
            0: 'Poor',
            1: 'Average',
            2: 'Good'
        }.get(int(label), 'Poor')

    def predict_single_student(self, data):
        """
        Predict performance for a single student

        Args:
            data (dict): Student feature values

        Returns:
            dict: Prediction result with probability and predicted marks
        """
        student_data = pd.DataFrame([data])
        feature_df = self._ensure_feature_frame(student_data)
        student_scaled = self.scaler.transform(feature_df.values)

        prediction = self.model.predict(student_scaled)[0]
        probability = self.model.predict_proba(student_scaled)[0]
        probability_map = dict(zip(self.model.classes_, probability))

        predicted_marks = None
        if self.regression_model is not None:
            try:
                predicted_marks = float(self.regression_model.predict(student_data)[0])
            except Exception:
                predicted_marks = None

        category = self._get_category_from_label(prediction)
        if predicted_marks is not None:
            category = self._get_category_from_marks(predicted_marks)

        return {
            'Performance': category,
            'Poor_Performance_Probability': probability_map.get(0, 0) * 100,
            'Average_Performance_Probability': probability_map.get(1, 0) * 100,
            'Good_Performance_Probability': probability_map.get(2, 0) * 100
        }

    def predict_batch(self, data_df):
        """
        Predict performance for multiple students

        Args:
            data_df (pd.DataFrame): DataFrame with student data

        Returns:
            pd.DataFrame: DataFrame with predictions
        """
        feature_df = self._ensure_feature_frame(data_df)
        data_scaled = self.scaler.transform(feature_df.values)

        predictions = self.model.predict(data_scaled)
        probabilities = self.model.predict_proba(data_scaled)

        result_df = data_df.copy()
        result_df['Classifier_Prediction'] = [self._get_category_from_label(p) for p in predictions]
        result_df['Confidence'] = np.max(probabilities, axis=1) * 100
        result_df['Poor_Performance_Probability'] = probabilities[:, list(self.model.classes_).index(0)] * 100 if 0 in self.model.classes_ else 0
        result_df['Average_Performance_Probability'] = probabilities[:, list(self.model.classes_).index(1)] * 100 if 1 in self.model.classes_ else 0
        result_df['Good_Performance_Probability'] = probabilities[:, list(self.model.classes_).index(2)] * 100 if 2 in self.model.classes_ else 0

        if self.regression_model is not None:
            try:
                result_df['Predicted_Marks'] = self.regression_model.predict(data_df)
            except Exception:
                result_df['Predicted_Marks'] = np.nan
        else:
            result_df['Predicted_Marks'] = np.nan

        result_df['Predicted_Performance'] = result_df.apply(
            lambda row: self._get_category_from_marks(row['Predicted_Marks'])
            if pd.notna(row['Predicted_Marks'])
            else self._get_category_from_label(
                self.model.classes_[np.argmax([row.get('Poor_Performance_Probability', 0), row.get('Average_Performance_Probability', 0), row.get('Good_Performance_Probability', 0)])]
            ),
            axis=1
        )

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
        weak_students = predictions_df[predictions_df['Predicted_Performance'] == 'Poor']
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

        category_counts = predictions_df['Predicted_Performance'].value_counts().to_dict()
        good_students = int(category_counts.get('Good', 0))
        average_students = int(category_counts.get('Average', 0))
        poor_students = int(category_counts.get('Poor', 0))

        report = {
            'total_students': len(predictions_df),
            'good_performance': good_students,
            'average_performance': average_students,
            'poor_performance': poor_students,
            'good_performance_percentage': (good_students / len(predictions_df)) * 100 if len(predictions_df) else 0,
            'average_performance_percentage': (average_students / len(predictions_df)) * 100 if len(predictions_df) else 0,
            'poor_performance_percentage': (poor_students / len(predictions_df)) * 100 if len(predictions_df) else 0,
            'average_confidence': float(predictions_df['Confidence'].mean()),
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
        if result.get('Predicted_Marks') is not None:
            print(f"Predicted Next Exam Marks: {result['Predicted_Marks']:.2f}")
        print("="*50 + "\n")

    def get_top_students(self, data_df, n=10, by='Predicted_Marks'):
        """
        Return top N students by a selected metric (probability or predicted marks).
        """
        df = self.predict_batch(data_df)
        if by not in df.columns:
            by = 'Predicted_Marks'
        return df.sort_values(by, ascending=False).head(n)

    def get_bottom_students(self, data_df, n=10, by='Predicted_Marks'):
        """
        Return bottom N students by a selected metric.
        """
        df = self.predict_batch(data_df)
        if by not in df.columns:
            by = 'Predicted_Marks'
        return df.sort_values(by, ascending=True).head(n)

    def class_performance_analytics(self, data_df, include_predictions=True):
        """
        Compute class-level analytics: means, medians, stds and performance breakdown.
        """
        df = data_df.copy()
        if include_predictions:
            df = self.predict_batch(df)

        numeric = df.select_dtypes(include=['number'])
        analytics = {
            'count': int(len(df)),
            'feature_means': numeric.mean().to_dict(),
            'feature_medians': numeric.median().to_dict(),
            'feature_stds': numeric.std().to_dict(),
        }

        if 'Predicted_Performance' in df.columns:
            analytics['performance_counts'] = df['Predicted_Performance'].value_counts().to_dict()
        if 'Confidence' in df.columns:
            analytics['average_confidence'] = float(df['Confidence'].mean())

        return analytics

    def attach_predicted_marks(self, df, regression_model):
        """
        If a regression model is provided, attach a `Predicted_Marks` column to df.
        """
        if regression_model is None:
            return df
        try:
            preds = regression_model.predict(df)
            df = df.copy()
            df['Predicted_Marks'] = preds
            return df
        except Exception:
            return df