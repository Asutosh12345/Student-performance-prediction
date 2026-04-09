# Student Performance Prediction System using Machine Learning

A comprehensive machine learning system that predicts student academic performance and helps teachers identify students who may need improvement. The system uses Logistic Regression and provides both a command-line interface and a web-based interface.

## Overview

This system analyzes student academic data and predicts whether a student will have **Good Performance** or **Poor Performance**. It helps teachers:
- Identify struggling students early
- Provide targeted interventions
- Monitor student progress
- Make data-driven decisions

## Features

### Core Features
- ✅ **Single Student Prediction** - Quick prediction for individual students
- ✅ **Batch Predictions** - Upload CSV files for multiple student predictions
- ✅ **Model Dashboard** - View model statistics and feature importance
- ✅ **REST API** - Complete REST API for predictions
- ✅ **Web Interface** - Beautiful, responsive web interface
- ✅ **Model Persistence** - Save and load trained models
- ✅ **Performance Metrics** - Comprehensive model evaluation metrics

### Technical Features
- ✅ Data preprocessing and normalization
- ✅ Feature scaling using StandardScaler
- ✅ Train-test split validation
- ✅ Model coefficient analysis
- ✅ Confusion matrix visualization
- ✅ Confidence scoring

## Dataset

The system analyzes 5 key academic features:

| Feature | Range | Description |
|---------|-------|-------------|
| **Attendance** | 0-100% | Class participation percentage |
| **Internal Marks** | 0-50 | Mid-term examination score |
| **Assignment Marks** | 0-20 | Assignment completion score |
| **Study Hours** | 0-10 | Weekly study hours |
| **Previous Results** | 0-100 | Historical academic performance |

**Output**: Good Performance / Poor Performance (Binary Classification)

## Model Performance

| Metric | Value |
|--------|-------|
| **Accuracy** | 90% |
| **Precision** | 92.31% |
| **Recall** | 92.31% |
| **F1-Score** | 92.31% |
| **ROC-AUC** | 93.13% |

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)

### Step 1: Clone or Download the Project
```bash
cd c:\Users\ASUS\Desktop\MCAProject
```

### Step 2: Create Virtual Environment
```bash
python -m venv .venv
.venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

## Usage

### Option 1: Web Interface

#### Start the Flask Server
```bash
python app.py
```

The application will be available at `http://localhost:5000`

#### Features
1. **Home Page** - Project overview and features
2. **Dashboard** - Model statistics and feature importance
3. **Predict Page** - Single and batch predictions

### Option 2: Command-Line Interface

#### Generate Dataset
```bash
python data/generate_data.py
```

#### Train Model and Make Predictions
```bash
python main.py
```

This will:
1. Generate/load student dataset
2. Preprocess the data
3. Train the model
4. Evaluate performance
5. Make sample predictions
6. Generate visualizations

## Project Structure

```
MCAProject/
├── app.py                          # Flask web application
├── main.py                         # Command-line interface
├── requirements.txt                # Python dependencies
├── README.md                       # This file
│
├── data/
│   ├── generate_data.py           # Dataset generation script
│   └── student_data.csv           # Generated dataset
│
├── src/
│   ├── data_preprocessing.py      # Data preprocessing module
│   ├── model_training.py          # Model training module
│   ├── prediction.py              # Prediction engine
│   └── visualization.py           # Visualization module
│
├── models/
│   ├── model.pkl                  # Trained model (serialized)
│   └── scaler.pkl                 # Feature scaler (serialized)
│
├── templates/
│   ├── base.html                  # Base template
│   ├── index.html                 # Home page
│   ├── predict.html               # Prediction page
│   ├── dashboard.html             # Dashboard page
│   └── 404.html                   # Error page
│
├── static/
│   ├── style.css                  # CSS styling
│   └── script.js                  # JavaScript utilities
│
└── output/
    ├── data_distribution.png      # Data visualization
    └── correlation_matrix.png     # Correlation heatmap
```

## API Endpoints

### 1. Single Prediction
**POST** `/api/predict`

Request:
```json
{
  "attendance": 85.5,
  "internal_marks": 42.0,
  "assignment_marks": 18.0,
  "study_hours": 8.0,
  "previous_results": 78.0
}
```

Response:
```json
{
  "status": "success",
  "performance": "Good Performance",
  "confidence": 99.99,
  "good_performance_prob": 99.99,
  "poor_performance_prob": 0.01
}
```

### 2. Batch Prediction
**POST** `/api/batch-predict`

- Upload CSV file with student data
- Returns predictions for all records

### 3. Model Statistics
**GET** `/api/model-stats`

Returns model type, features, and status

### 4. Feature Importance
**GET** `/api/feature-importance`

Returns model coefficients for each feature

### 5. Download Template
**GET** `/api/download-template`

Downloads CSV template for batch predictions

### 6. Health Check
**GET** `/health`

Returns system health status

## Batch Prediction CSV Format

**Required Columns:**
```csv
Attendance,Internal_Marks,Assignment_Marks,Study_Hours,Previous_Results
85.5,42.0,18.0,8.0,78.0
92.3,38.5,19.5,6.5,85.0
78.1,25.3,15.2,4.0,65.0
```

## Feature Importance

The model coefficients show the importance of each feature:

| Feature | Coefficient | Importance |
|---------|-------------|------------|
| Internal Marks | 2.3528 | Highest |
| Study Hours | 1.9136 | High |
| Assignment Marks | 1.4238 | High |
| Attendance | 1.0929 | Medium |
| Previous Results | 0.8125 | Medium |

## Configuration

### Model Configuration
Edit in `src/model_training.py`:
```python
# Model parameters
model = LogisticRegression(
    random_state=42,
    max_iter=1000,
    solver='lbfgs'
)
```

### Flask Configuration
Edit in `app.py`:
```python
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Max upload size
app.run(
    host='0.0.0.0',
    port=5000,
    debug=True
)
```

## Troubleshooting

### Issue: Module not found errors
**Solution:** Ensure virtual environment is activated and all dependencies are installed
```bash
pip install -r requirements.txt
```

### Issue: Port 5000 already in use
**Solution:** Change the port in `app.py`:
```python
app.run(port=5001)
```

### Issue: CSV file not accepted
**Solution:** Ensure CSV has required columns and correct format

## Performance Optimization

### For Batch Predictions
- Process large CSV files (tested up to 10,000 records)
- Use Feature Importance analysis for model optimization

### Model Improvement
- Collect more training data
- Feature engineering
- Hyperparameter tuning
- Cross-validation

## Deployment

### Local Deployment
```bash
python app.py
```

### Production Deployment
Install production server:
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### Docker Deployment
Create `Dockerfile`:
```dockerfile
FROM python:3.9
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]
```

Build and run:
```bash
docker build -t student-predictor .
docker run -p 5000:5000 student-predictor
```

## Testing

### Test Single Prediction
```python
from src.prediction import PredictionEngine
import pickle

# Load model
with open('models/model.pkl', 'rb') as f:
    model = pickle.load(f)

# Make prediction
import numpy as np
features = np.array([[85, 42, 18, 8, 78]])
prediction = model.predict(features)
```

### Test API
```bash
# Test single prediction
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"attendance":85.5,"internal_marks":42.0,"assignment_marks":18.0,"study_hours":8.0,"previous_results":78.0}'

# Test health
curl http://localhost:5000/health
```

## Future Enhancements

- [ ] Multi-class classification (Excellent/Good/Average/Poor)
- [ ] Time-series analysis for student trends
- [ ] Advanced visualizations with Plotly
- [ ] Student recommendation system
- [ ] Integration with Student Information System (SIS)
- [ ] Mobile app support
- [ ] Real-time notifications
- [ ] Advanced reporting features
- [ ] Teacher analytics dashboard
- [ ] Student progress tracking

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## License

This project is open source and available under the MIT License.

## Contact & Support

For questions or issues, please contact:
- Project Developer: AI Assistant
- Created: March 2026

## Acknowledgments

- Machine Learning with Scikit-learn
- Data Analysis with Pandas and NumPy
- Visualization with Matplotlib and Seaborn
- Web Framework: Flask

---

**Note:** This system is designed for educational purposes. Always validate predictions with domain experts before making critical decisions.
