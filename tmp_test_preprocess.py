from data.generate_data import generate_student_data
from src.data_preprocessing import DataPreprocessor

if __name__ == '__main__':
    df = generate_student_data(n_samples=10)
    print(df.head())
    print(df.columns.tolist())
    df.to_csv('tmp_student_data.csv', index=False)
    preprocessor = DataPreprocessor('tmp_student_data.csv')
    X_train, X_test, y_train, y_test = preprocessor.preprocess(test_size=0.2, random_state=42)
    print('X_train shape', X_train.shape)
    print('y_train distribution', y_train.value_counts().to_dict())
