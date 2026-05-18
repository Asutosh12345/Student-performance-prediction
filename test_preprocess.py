import sys
sys.path.insert(0, 'src')
import os
from data.generate_data import generate_student_data
from data_preprocessing import DataPreprocessor

# Generate data
df = generate_student_data(200)
os.makedirs('data', exist_ok=True)
df.to_csv('data/student_data.csv', index=False)
print("Data generated")

# Try preprocessing
p = DataPreprocessor('data/student_data.csv')
X_train, X_test, y_train, y_test = p.preprocess()
print(f'X_train shape: {X_train.shape}')
print(f'Performance value counts:\n{p.df["Performance"].value_counts()}')
print(f'y_train unique: {y_train}')
print(f'y_train unique values: {set(y_train)}')
