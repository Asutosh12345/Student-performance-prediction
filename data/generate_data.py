import pandas as pd
import numpy as np
from sklearn.datasets import make_classification

def generate_student_data(n_samples=200, random_state=42):
    """
    Generate synthetic student performance dataset
    
    Features:
    - Attendance (0-100%)
    - Internal marks (0-50)
    - Assignment marks (0-20)
    - Study hours (0-10 per week)
    - Previous results (0-100)
    
    Target:
    - Performance: Good (1) or Poor (0)
    """
    np.random.seed(random_state)
    
    # Generate synthetic data
    n_samples = n_samples
    
    attendance = np.random.uniform(40, 100, n_samples)
    internal_marks = np.random.uniform(0, 50, n_samples)
    assignment_marks = np.random.uniform(0, 20, n_samples)
    study_hours = np.random.uniform(0, 10, n_samples)
    previous_results = np.random.uniform(20, 100, n_samples)
    
    # Create performance label based on weighted features
    # Students with better attendance, marks, and study hours perform better
    performance_score = (
        (attendance / 100) * 0.25 +
        (internal_marks / 50) * 0.25 +
        (assignment_marks / 20) * 0.15 +
        (study_hours / 10) * 0.20 +
        (previous_results / 100) * 0.15
    )
    
    # Add some noise
    performance_score += np.random.normal(0, 0.05, n_samples)
    
    # Binary classification: Good (1) if score > 0.5, Poor (0) otherwise
    performance = (performance_score > 0.5).astype(int)
    
    # Create DataFrame
    df = pd.DataFrame({
        'Attendance': np.round(attendance, 2),
        'Internal_Marks': np.round(internal_marks, 2),
        'Assignment_Marks': np.round(assignment_marks, 2),
        'Study_Hours': np.round(study_hours, 2),
        'Previous_Results': np.round(previous_results, 2),
        'Performance': performance
    })
    
    return df

if __name__ == "__main__":
    # Generate and save dataset
    df = generate_student_data(n_samples=200)
    df.to_csv('student_data.csv', index=False)
    print("Dataset generated successfully!")
    print(f"\nDataset shape: {df.shape}")
    print(f"\nFirst few rows:\n{df.head()}")
    print(f"\nDataset info:\n{df.info()}")
    print(f"\nPerformance distribution:\n{df['Performance'].value_counts()}")
