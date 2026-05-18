"""
Student Performance Prediction System using Machine Learning

This system predicts student academic performance using logistic regression.
Features analyzed:
- Attendance (%)
- Internal Marks
- Previous Grade Value

Output: 3-class prediction — Poor / Average / Good performance
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data_preprocessing import DataPreprocessor
from model_training import StudentPerformanceModel
from prediction import PredictionEngine
from visualization import VisualizationEngine
import pandas as pd
import numpy as np


def main():
    print("=" * 60)
    print("STUDENT PERFORMANCE PREDICTION SYSTEM")
    print("Using Logistic Regression Machine Learning Model")
    print("=" * 60)

    # ------------------------------------------------------------------ #
    # STEP 1 — Dataset
    # ------------------------------------------------------------------ #
    print("\n[STEP 1] Preparing Dataset...")
    data_path = os.path.join(os.path.dirname(__file__), 'data', 'student_data.csv')

    if not os.path.exists(data_path):
        print("Generating student dataset...")
        from generate_data import generate_student_data
        df = generate_student_data(n_samples=200)
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        df.to_csv(data_path, index=False)
        print(f"Dataset saved to {data_path}")
    else:
        print(f"Loading existing dataset from {data_path}")

    # ------------------------------------------------------------------ #
    # STEP 2 — Preprocessing
    # ------------------------------------------------------------------ #
    print("\n[STEP 2] Data Preprocessing...")
    preprocessor = DataPreprocessor(data_path)
    X_train, X_test, y_train, y_test = preprocessor.preprocess()

    # FIX 13: feature_names must match DataPreprocessor.FEATURE_COLUMNS exactly.
    # Old main.py had 6 names including 'Pass_Fail' which is not in FEATURE_COLUMNS
    # and is dropped during preprocessing, so scaled arrays only had 5 (or 3)
    # columns — causing shape mismatches at prediction time.
    feature_names = DataPreprocessor.FEATURE_COLUMNS  # single source of truth

    # ------------------------------------------------------------------ #
    # STEP 3 — Training
    # ------------------------------------------------------------------ #
    print("\n[STEP 3] Training Logistic Regression Model...")
    model = StudentPerformanceModel()
    model.train(X_train, y_train)

    # ------------------------------------------------------------------ #
    # STEP 4 — Evaluation
    # ------------------------------------------------------------------ #
    print("\n[STEP 4] Evaluating Model Performance...")
    metrics = model.evaluate(X_test, y_test)

    # ------------------------------------------------------------------ #
    # STEP 5 — Feature importance
    # ------------------------------------------------------------------ #
    print("\n[STEP 5] Feature Importance Analysis...")
    # FIX 14: Pass the full coef_ matrix (n_classes × n_features); visualization
    # now averages across classes.  Old code passed coef_[0] — one class only.
    model.get_model_coefficients(feature_names)

    # ------------------------------------------------------------------ #
    # STEP 6 — Prediction engine
    # ------------------------------------------------------------------ #
    print("\n[STEP 6] Setting up Prediction Engine...")
    prediction_engine = PredictionEngine(model.model, preprocessor.scaler, feature_names)

    # ------------------------------------------------------------------ #
    # STEP 7 — Single & batch predictions
    # ------------------------------------------------------------------ #
    print("\n[STEP 7] Making Predictions on Test Students...")

    print("\nExample 1: Single Student Prediction")
    print("-" * 50)
    result = prediction_engine.predict_single_student({
        'Attendance':           85.0,
        'Internal_Marks':       40.0,
        'Previous_Grade_Value': 80.0,
    })
    prediction_engine.print_prediction_result(result)

    print("\nExample 2: Batch Predictions on Test Data")
    print("-" * 50)
    # FIX 17: X_test is already scaled — passing it to predict_batch would scale
    # it a second time, producing garbage predictions.  Use the raw (unscaled)
    # feature values from the preprocessed DataFrame instead.
    raw_test_df = preprocessor.df[feature_names].iloc[y_test.index].reset_index(drop=True)
    predictions_df = prediction_engine.predict_batch(raw_test_df)
    predictions_df['Actual_Performance'] = y_test.values
    print("\nFirst 10 predictions:")
    display_cols = [c for c in feature_names if c in predictions_df.columns]
    display_cols += ['Predicted_Performance', 'Confidence']
    print(predictions_df[display_cols].head(10))

    # ------------------------------------------------------------------ #
    # STEP 8 — Weak student identification
    # ------------------------------------------------------------------ #
    print("\n[STEP 8] Identifying Students Needing Improvement...")
    print("-" * 50)
    weak_students = prediction_engine.identify_weak_students(raw_test_df, threshold=0.4)
    print(f"\nStudents predicted as 'Poor': {len(weak_students)}")
    if len(weak_students) > 0:
        show_cols = [c for c in feature_names if c in weak_students.columns]
        show_cols += ['Predicted_Performance', 'Good_Performance_Probability']
        print("\nTop 5 students needing improvement:")
        print(weak_students[show_cols].head(5))

    # ------------------------------------------------------------------ #
    # STEP 9 — Report
    # ------------------------------------------------------------------ #
    print("\n[STEP 9] Generating Performance Report...")
    print("-" * 50)
    report = prediction_engine.generate_report(raw_test_df)
    print(f"\nPerformance Summary:")
    print(f"Total Students Analysed : {report['total_students']}")
    print(f"Good    Performance     : {report['good_performance']}    ({report['good_performance_percentage']:.2f}%)")
    print(f"Average Performance     : {report['average_performance']} ({report['average_performance_percentage']:.2f}%)")
    print(f"Poor    Performance     : {report['poor_performance']}    ({report['poor_performance_percentage']:.2f}%)")
    print(f"Avg Prediction Confidence: {report['average_confidence']:.2f}%")

    # ------------------------------------------------------------------ #
    # STEP 10 — Visualisations
    # ------------------------------------------------------------------ #
    print("\n[STEP 10] Generating Visualizations...")
    print("-" * 50)
    viz = VisualizationEngine()
    original_df = preprocessor.df  # already-preprocessed frame with Performance column

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

    print("5. ROC curves...")
    # FIX 15: pass the full probability matrix — visualization handles OvR internally
    viz.plot_roc_curve(y_test, model.y_pred_proba,
                       save_path=os.path.join(output_dir, 'roc_curve.png'))

    print("6. Feature importance...")
    # FIX 16: pass full coef_ matrix, not just coef_[0]
    viz.plot_feature_importance(model.model.coef_, feature_names,
                                save_path=os.path.join(output_dir, 'feature_importance.png'))

    print("7. Predictions distribution...")
    viz.plot_predictions_distribution(predictions_df,
                                      save_path=os.path.join(output_dir, 'predictions_distribution.png'))

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE!")
    print("=" * 60)
    print("\nVisualization files saved to:", output_dir)
    print("\nKey Findings:")
    print(f"  Model Accuracy  : {metrics['accuracy']  * 100:.2f}%")
    print(f"  Model Precision : {metrics['precision'] * 100:.2f}%")
    print(f"  Model Recall    : {metrics['recall']    * 100:.2f}%")
    print(f"  ROC-AUC Score   : {metrics['roc_auc']:.4f}")
    print(f"\nThe system identified {len(weak_students)} students who may need extra support.")

    return model, prediction_engine, preprocessor


if __name__ == "__main__":
    model, pred_engine, prep = main()