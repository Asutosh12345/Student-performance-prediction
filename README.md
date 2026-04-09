# Student Performance Prediction System

A machine learning system that predicts student academic performance using Logistic Regression.

## Overview

This project analyzes student academic data and predicts whether students will have good or poor performance. It helps teachers identify students who may need additional support early in the academic term.

## Features

- **Data Collection & Preprocessing**: Handles student data loading, validation, and standardization
- **Machine Learning Model**: Logistic Regression classifier for binary performance classification
- **Prediction Engine**: Predicts performance for individual students or batches
- **Weak Student Identification**: Automatically identifies students who need improvement
- **Comprehensive Analysis**: Generates detailed reports and visualizations
- **Model Evaluation**: Provides multiple performance metrics (Accuracy, Precision, Recall, F1-Score, ROC-AUC)

## Dataset

The system analyzes the following student attributes:

| Feature | Range | Description |
|---------|-------|-------------|
| Attendance | 0-100 | Attendance percentage |
| Internal Marks | 0-50 | Internal assessment marks |
| Assignment Marks | 0-20 | Assignment submission marks |
| Study Hours | 0-10 | Study hours per week |
| Previous Results | 0-100 | Previous academic performance |

**Target Variable**: 
- Good Performance (1)
- Poor Performance (0)

## Project Structure

```
MCAProject/
├── data/
│   ├── student_data.csv          # Student dataset
│   └── generate_data.py          # Dataset generation script
├── src/
│   ├── data_preprocessing.py     # Data loading and preprocessing
│   ├── model_training.py         # Logistic Regression model
│   ├── prediction.py             # Prediction engine
│   └── visualization.py          # Visualization tools
├── output/                       # Generated visualizations
├── main.py                       # Main execution script
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Installation

### Prerequisites
- Python 3.7+
- pip (Python package manager)

### Setup

1. **Navigate to project directory**:
```bash
cd c:\Users\ASUS\Desktop\MCAProject
```

2. **Create a virtual environment (optional but recommended)**:
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

3. **Install required packages**:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Complete System

Execute the main script to run the entire pipeline:

```bash
python main.py
```

This will:
1. Generate or load the student dataset
2. Preprocess and prepare the data
3. Train the Logistic Regression model
4. Evaluate model performance
5. Make predictions on test data
6. Generate visualizations
7. Identify weak students
8. Display comprehensive analysis report

### Using Individual Components

You can also use individual components in your own scripts:

```python
from src.data_preprocessing import DataPreprocessor
from src.model_training import StudentPerformanceModel
from src.prediction import PredictionEngine

# 1. Data Preprocessing
preprocessor = DataPreprocessor('data/student_data.csv')
X_train, X_test, y_train, y_test = preprocessor.preprocess()

# 2. Model Training
model = StudentPerformanceModel()
model.train(X_train, y_train)
metrics = model.evaluate(X_test, y_test)

# 3. Make Predictions
pred_engine = PredictionEngine(model.model, preprocessor.scaler)
result = pred_engine.predict_single_student(
    attendance=85,
    internal_marks=40,
    assignment_marks=18,
    study_hours=7,
    previous_results=75
)
print(result)
```

## Model Information

### Algorithm
**Logistic Regression** is used for binary classification due to:
- Interpretability: Easy to understand feature importance
- Efficiency: Fast training and prediction
- Robustness: Works well with normalized features
- Probabilistic output: Provides confidence scores

### Model Training
- **Training-Test Split**: 80% training, 20% testing
- **Feature Scaling**: StandardScaler normalization
- **Regularization**: L2 regularization (default)
- **Max Iterations**: 1000

## Performance Metrics

The model reports the following evaluation metrics:

| Metric | Description |
|--------|-------------|
| Accuracy | Percentage of correct predictions |
| Precision | Accuracy of positive predictions |
| Recall | Coverage of actual positive cases |
| F1-Score | Harmonic mean of precision and recall |
| ROC-AUC | Area under ROC curve (0-1 scale) |
| Confusion Matrix | True/False Positives and Negatives |

## Visualizations

The system generates the following visualization charts:

1. **Data Distribution** - Histogram of all input features
2. **Correlation Matrix** - Heatmap of feature correlations
3. **Performance Distribution** - Bar chart of good vs poor performance
4. **Confusion Matrix** - Model prediction accuracy breakdown
5. **ROC Curve** - Receiver Operating Characteristic curve
6. **Feature Importance** - Coefficient values for each feature
7. **Predictions Distribution** - Pie chart and confidence histogram

All visualizations are saved in the `output/` directory as PNG files.

## Example Output

### Single Student Prediction
```
==================================================
PREDICTION RESULT
==================================================
Performance: Good
Confidence: 87.34%
Good Performance Probability: 87.34%
Poor Performance Probability: 12.66%
==================================================
```

### Model Metrics
```
==================================================
MODEL EVALUATION METRICS
==================================================
Accuracy:  0.8750
Precision: 0.8889
Recall:    0.8421
F1-Score:  0.8649
ROC-AUC:   0.9245
```

## Weak Student Identification

The system automatically identifies students with predicted good performance probability below a threshold (default: 40%). These students are flagged as needing improvement and can be provided with targeted support.

## Dependencies

- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning algorithms
- **matplotlib**: Visualization library
- **seaborn**: Statistical data visualization

## Results and Conclusions

### System Capabilities
✓ Accurately predicts student performance with high confidence  
✓ Identifies key factors affecting academic success  
✓ Flags at-risk students early for intervention  
✓ Provides probabilistic confidence scores  
✓ Generates comprehensive analysis reports  

### Benefits for Educators
- **Early Intervention**: Identify struggling students before they fall behind
- **Data-Driven Decisions**: Make informed decisions based on analytics
- **Personalized Support**: Target help to students who need it most
- **Performance Tracking**: Monitor student progress over time
- **Resource Optimization**: Allocate resources where they're needed

## Future Enhancements

Potential improvements to the system:

1. **Advanced Models**: Try Random Forest, SVM, or Neural Networks
2. **Feature Engineering**: Create derived features from raw data
3. **Cross-Validation**: Implement k-fold cross-validation
4. **Hyperparameter Tuning**: Optimize model parameters
5. **Time Series Analysis**: Track performance trends over time
6. **Interactive Dashboard**: Web-based UI for predictions and monitoring
7. **Integration**: Connect with student information systems
8. **Explainability**: Implement SHAP or LIME for model interpretability

## License

This project is provided for educational and research purposes.

## Author

Student Performance Prediction System  
Created for Machine Learning Education

## Support

For issues, questions, or suggestions, please refer to the project documentation or contact the development team.

---

**Last Updated**: March 2026  
**Status**: Complete and Functional
