

import os

# Project directories
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'output')

# File paths
DATA_FILE = os.path.join(DATA_DIR, 'student_data.csv')
MODEL_FILE = os.path.join(MODELS_DIR, 'model.pkl')
SCALER_FILE = os.path.join(MODELS_DIR, 'scaler.pkl')

# Flask configuration
FLASK_DEBUG = True
FLASK_PORT = 5000
FLASK_HOST = '0.0.0.0'
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB

# Model configuration
RANDOM_STATE = 42
TEST_SIZE = 0.2
TRAIN_SAMPLES = 200

# Model parameters
MODEL_PARAMS = {
    'random_state': RANDOM_STATE,
    'max_iter': 1000,
    'solver': 'lbfgs'
}

# Feature names
FEATURES = [
    'Attendance',
    'Internal_Marks',
    'Assignment_Marks',
    'Study_Hours',
    'Previous_Results'
]

# Feature ranges
FEATURE_RANGES = {
    'Attendance': (0, 100),
    'Internal_Marks': (0, 50),
    'Assignment_Marks': (0, 20),
    'Study_Hours': (0, 10),
    'Previous_Results': (0, 100)
}

# Model metrics
EXPECTED_ACCURACY = 0.90

# Logging
LOG_LEVEL = 'INFO'
LOG_FILE = os.path.join(PROJECT_ROOT, 'app.log')

# Create directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
