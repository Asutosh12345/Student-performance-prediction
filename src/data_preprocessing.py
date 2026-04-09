import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class DataPreprocessor:
    """
    Handles data loading, cleaning, and preprocessing for student performance prediction
    """
    
    def __init__(self, filepath):
        """
        Initialize the preprocessor with data file path
        
        Args:
            filepath (str): Path to the CSV data file
        """
        self.filepath = filepath
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        
    def load_data(self):
        """Load data from CSV file"""
        try:
            self.df = pd.read_csv(self.filepath)
            print(f"Data loaded successfully! Shape: {self.df.shape}")
            return self.df
        except FileNotFoundError:
            print(f"Error: File {self.filepath} not found!")
            return None
    
    def display_data_info(self):
        """Display basic information about the dataset"""
        if self.df is None:
            print("No data loaded. Please load data first.")
            return
        
        print("\n" + "="*50)
        print("DATASET INFORMATION")
        print("="*50)
        print(f"\nDataset Shape: {self.df.shape}")
        print(f"\nFirst 5 rows:\n{self.df.head()}")
        print(f"\nData Types:\n{self.df.dtypes}")
        print(f"\nMissing Values:\n{self.df.isnull().sum()}")
        print(f"\nStatistical Summary:\n{self.df.describe()}")
    
    def check_missing_values(self):
        """Check and handle missing values"""
        if self.df is None:
            print("No data loaded.")
            return
        
        missing = self.df.isnull().sum()
        if missing.sum() == 0:
            print("No missing values found!")
            return True
        else:
            print(f"Missing values:\n{missing[missing > 0]}")
            # Fill missing values with mean for numeric columns
            numeric_columns = self.df.select_dtypes(include=[np.number]).columns
            self.df[numeric_columns] = self.df[numeric_columns].fillna(self.df[numeric_columns].mean())
            print("Missing values filled with mean.")
            return True
    
    def split_data(self, test_size=0.2, random_state=42):
        """
        Split data into training and testing sets
        
        Args:
            test_size (float): Proportion of test set (default: 0.2)
            random_state (int): Random seed for reproducibility
        """
        if self.df is None:
            print("No data loaded.")
            return
        
        # Separate features and target
        X = self.df.drop('Performance', axis=1)
        y = self.df['Performance']
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"\nData split completed:")
        print(f"Training set: {self.X_train.shape}")
        print(f"Testing set: {self.X_test.shape}")
        print(f"Training performance distribution:\n{self.y_train.value_counts()}")
        
    def scale_features(self):
        """Scale features using StandardScaler"""
        if self.X_train is None:
            print("Data not split yet. Please split data first.")
            return
        
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        
        print("\nFeatures scaled using StandardScaler")
        
    def get_processed_data(self):
        """Return processed training and testing data"""
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def preprocess(self, test_size=0.2, random_state=42):
        """
        Complete preprocessing pipeline
        
        Args:
            test_size (float): Proportion of test set
            random_state (int): Random seed
        """
        self.load_data()
        self.display_data_info()
        self.check_missing_values()
        self.split_data(test_size, random_state)
        self.scale_features()
        
        return self.get_processed_data()
