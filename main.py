"""
Student Performance Prediction System using Machine Learning

This system predicts student academic performance using logistic regression.
Features analyzed:
- Attendance
- Internal marks
- Assignment marks
- Study hours
- Previous results

Output: Prediction of Good or Poor performance
"""

import os
import sys

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data_preprocessing import DataPreprocessor
from model_training import StudentPerformanceModel
from prediction import PredictionEngine
from visualization import VisualizationEngine
import pandas as pd

def main():
    """Main execution function"""
    
    print("="*60)
    print("STUDENT PERFORMANCE PREDICTION SYSTEM")
    print("Using Logistic Regression Machine Learning Model")
    print("="*60)
    
    # 1. Generate Dataset (if doesn't exist)
    print("\n[STEP 1] Preparing Dataset...")
    data_path = os.path.join(os.path.dirname(__file__), 'data', 'student_data.csv')
    
    if not os.path.exists(data_path):
        print("Generating student dataset...")
        from data.generate_data import generate_student_data
        df = generate_student_data(n_samples=200)
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        df.to_csv(data_path, index=False)
        print(f"Dataset saved to {data_path}")
    else:
        print(f"Loading existing dataset from {data_path}")
    
    # 2. Data Preprocessing
    print("\n[STEP 2] Data Preprocessing...")
    preprocessor = DataPreprocessor(data_path)
    X_train, X_test, y_train, y_test = preprocessor.preprocess()
    
    # 3. Model Training
    print("\n[STEP 3] Training Logistic Regression Model...")
    model = StudentPerformanceModel()
    model.train(X_train, y_train)
    
    # 4. Model Evaluation
    print("\n[STEP 4] Evaluating Model Performance...")
    metrics = model.evaluate(X_test, y_test)
    
    # 5. Display Feature Importance
    print("\n[STEP 5] Feature Importance Analysis...")
    feature_names = ['Attendance', 'Internal_Marks', 'Assignment_Marks', 'Study_Hours', 'Previous_Results']
    model.get_model_coefficients(feature_names)
    
    # 6. Create Prediction Engine
    print("\n[STEP 6] Setting up Prediction Engine...")
    prediction_engine = PredictionEngine(model.model, preprocessor.scaler, feature_names)
    
    # 7. Make Predictions on New Students
    print("\n[STEP 7] Making Predictions on Test Students...")
    
    # Example 1: Single Student Prediction
    print("\nExample 1: Single Student Prediction")
    print("-" * 50)
    result = prediction_engine.predict_single_student(
        attendance=85,
        internal_marks=40,
        assignment_marks=18,
        study_hours=7,
        previous_results=75
    )
    prediction_engine.print_prediction_result(result)
    
    # Example 2: Batch Predictions
    print("\nExample 2: Batch Predictions on Test Data")
    print("-" * 50)
    test_df = pd.DataFrame(X_test, columns=feature_names)
    
    predictions_df = prediction_engine.predict_batch(test_df)
    predictions_df['Actual_Performance'] = y_test.values
    print("\nFirst 10 predictions:")
    print(predictions_df[['Attendance', 'Internal_Marks', 'Assignment_Marks', 
                          'Study_Hours', 'Previous_Results', 'Predicted_Performance', 
                          'Confidence']].head(10))
    
    # 8. Identify Weak Students
    print("\n[STEP 8] Identifying Students Needing Improvement...")
    print("-" * 50)
    weak_students = prediction_engine.identify_weak_students(test_df, threshold=0.4)
    print(f"\nNumber of students needing improvement (probability < 40%): {len(weak_students)}")
    if len(weak_students) > 0:
        print("\nTop 5 students needing improvement:")
        print(weak_students[['Attendance', 'Internal_Marks', 'Study_Hours', 
                            'Predicted_Performance', 'Good_Performance_Probability']].head(5))
    
    # 9. Generate Report
    print("\n[STEP 9] Generating Performance Report...")
    print("-" * 50)
    report = prediction_engine.generate_report(test_df)
    
    print(f"\nPerformance Summary:")
    print(f"Total Students Analyzed: {report['total_students']}")
    print(f"Good Performance: {report['good_performance']} ({report['good_performance_percentage']:.2f}%)")
    print(f"Poor Performance: {report['poor_performance']} ({report['poor_performance_percentage']:.2f}%)")
    print(f"Average Prediction Confidence: {report['average_confidence']:.2f}%")
    
    # 10. Visualizations
    print("\n[STEP 10] Generating Visualizations...")
    print("-" * 50)
    
    viz = VisualizationEngine()
    
    # Load original data for visualizations
    original_df = pd.read_csv(data_path)
    
    print("Creating visualizations...")
    
    # These will be displayed/saved in the output directory
    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    print("1. Data distribution plot...")
    viz.plot_data_distribution(original_df, 
                               save_path=os.path.join(output_dir, 'data_distribution.png'))
    
    print("2. Correlation matrix...")
    viz.plot_correlation_matrix(original_df,
                               save_path=os.path.join(output_dir, 'correlation_matrix.png'))
    
    print("3. Performance distribution...")
    viz.plot_performance_distribution(original_df,
                                     save_path=os.path.join(output_dir, 'performance_distribution.png'))
    
    print("4. Confusion matrix...")
    viz.plot_confusion_matrix(metrics['confusion_matrix'],
                             save_path=os.path.join(output_dir, 'confusion_matrix.png'))
    
    print("5. ROC curve...")
    viz.plot_roc_curve(y_test, model.y_pred_proba,
                      save_path=os.path.join(output_dir, 'roc_curve.png'))
    
    print("6. Feature importance...")
    viz.plot_feature_importance(model.model.coef_[0], feature_names,
                               save_path=os.path.join(output_dir, 'feature_importance.png'))
    
    print("7. Predictions distribution...")
    viz.plot_predictions_distribution(predictions_df,
                                     save_path=os.path.join(output_dir, 'predictions_distribution.png'))
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60)
    print("\nVisualization files saved to:", output_dir)
    print("\nKey Findings:")
    print(f"✓ Model Accuracy: {metrics['accuracy']*100:.2f}%")
    print(f"✓ Model Precision: {metrics['precision']*100:.2f}%")
    print(f"✓ Model Recall: {metrics['recall']*100:.2f}%")
    print(f"✓ ROC-AUC Score: {metrics['roc_auc']:.4f}")
    print(f"\nThe system successfully predicts student performance and can help")
    print(f"teachers identify {len(weak_students)} students who need improvement.")
    
    return model, prediction_engine, preprocessor

if __name__ == "__main__":
    model, pred_engine, prep = main()
