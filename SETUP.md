# Setup and Installation Guide

## Quick Start

### For Web Interface Users:
```bash
# 1. Navigate to project directory
cd c:\Users\ASUS\Desktop\MCAProject

# 2. Activate virtual environment
.venv\Scripts\activate

# 3. Start the web application
python run.py

# 4. Open browser
# Go to: http://localhost:5000
```

### For Command Line Users:
```bash
# 1. Navigate to project directory
cd c:\Users\ASUS\Desktop\MCAProject

# 2. Activate virtual environment
.venv\Scripts\activate

# 3. Run the main script
python main.py
```

---

## Detailed Installation Steps

### Step 1: Prerequisites
- Windows 10/11
- Python 3.8 or higher
- 100MB free disk space
- Modern web browser (Chrome, Firefox, Edge)

### Step 2: Navigate to Project Directory
```bash
cd c:\Users\ASUS\Desktop\MCAProject
```

### Step 3: Activate Virtual Environment
```bash
# On Windows
.venv\Scripts\activate

# On Mac/Linux
source .venv/bin/activate
```

### Step 4: Install/Verify Dependencies
```bash
# Install or update all dependencies
pip install -r requirements.txt

# Or run auto-setup
python setup.py
```

### Step 5: Verify Installation
```bash
# Test imports
python -c "import flask, pandas, sklearn; print('✓ All dependencies installed')"
```

---

## Starting the Application

### Option 1: Web Interface (Recommended)

**Method A - Using run.py:**
```bash
python run.py
```

**Method B - Using app.py:**
```bash
python app.py
```

**Method C - Using Flask CLI:**
```bash
flask --app app run
```

**Expected Output:**
```
============================================================
STUDENT PERFORMANCE PREDICTION SYSTEM
Starting Flask Web Application
============================================================

✓ Server starting on http://0.0.0.0:5000
✓ Press Ctrl+C to stop the server

 * Running on http://127.0.0.1:5000
```

**Access the Application:**
- Open your web browser
- Go to: `http://localhost:5000`

### Option 2: Command Line Interface

```bash
python main.py
```

This will:
1. Generate/load student dataset
2. Preprocess the data
3. Train the Logistic Regression model
4. Display model performance metrics
5. Make sample predictions
6. Generate visualizations (data_distribution.png, correlation_matrix.png)

---

## Web Interface Features

### Home Page (`/`)
- Project overview
- Feature explanation
- Model performance metrics
- Quick links to prediction and dashboard

### Predict Page (`/predict`)

**Single Student Prediction:**
- Enter student data directly
- Get immediate prediction
- View confidence score and probabilities

**Batch Upload:**
- Upload CSV file with multiple students
- Download template for correct format
- View batch results in table format

### Dashboard (`/dashboard`)
- Model statistics
- Feature importance visualization
- Model metrics (Accuracy, Precision, Recall, etc.)
- Confusion matrix

---

## API Endpoints Reference

### 1. Single Prediction
```bash
POST /api/predict
Content-Type: application/json

{
  "attendance": 85.5,
  "internal_marks": 42.0,
  "assignment_marks": 18.0,
  "study_hours": 8.0,
  "previous_results": 78.0
}
```

### 2. Batch Prediction
```bash
POST /api/batch-predict
Content-Type: multipart/form-data

File: CSV with columns (Attendance, Internal_Marks, etc.)
```

### 3. Model Statistics
```bash
GET /api/model-stats
```

### 4. Feature Importance
```bash
GET /api/feature-importance
```

### 5. Download CSV Template
```bash
GET /api/download-template
```

### 6. Health Check
```bash
GET /health
```

---

## Project Structure

```
MCAProject/
├── app.py                          # Flask web application
├── main.py                         # Command-line interface
├── run.py                          # Flask run script
├── setup.py                        # Setup and verification script
├── config.py                       # Configuration file
├── requirements.txt                # Python dependencies
├── README.md                       # Original documentation
├── README_UPDATED.md              # Complete documentation
│
├── data/
│   ├── generate_data.py           # Dataset generation
│   └── student_data.csv           # Generated dataset
│
├── src/
│   ├── data_preprocessing.py      # Data preprocessing module
│   ├── model_training.py          # Model training module
│   ├── prediction.py              # Prediction engine
│   └── visualization.py           # Visualization module
│
├── models/
│   ├── model.pkl                  # Trained logistic regression model
│   └── scaler.pkl                 # StandardScaler object
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

---

## CSV Template Format

For batch predictions, use this format:

```csv
Attendance,Internal_Marks,Assignment_Marks,Study_Hours,Previous_Results
85.5,42.0,18.0,8.0,78.0
92.3,38.5,19.5,6.5,85.0
78.1,25.3,15.2,4.0,65.0
```

**Column Requirements:**
- `Attendance`: 0-100 (percentage)
- `Internal_Marks`: 0-50 (score)
- `Assignment_Marks`: 0-20 (score)
- `Study_Hours`: 0-10 (hours per week)
- `Previous_Results`: 0-100 (percentage)

**Download Template:**
- On Predict page, click "Download Template"
- Or access via: `/api/download-template`

---

## Troubleshooting

### Issue: "Module not found" errors
**Solution:**
```bash
# Activate virtual environment
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Issue: Port 5000 already in use
**Solution 1 - Change port in run.py:**
```python
app.run(port=5001)
```

**Solution 2 - Find and stop process:**
```bash
# Find process using port 5000
netstat -ano | findstr :5000

# Kill process (replace PID with actual process ID)
taskkill /PID <PID> /F
```

### Issue: Can't access http://localhost:5000
**Solution:**
1. Check Flask is running (look for console output)
2. Try http://127.0.0.1:5000 instead
3. Check firewall settings
4. Try restarting the application

### Issue: CSV file not accepted
**Solution:**
- Ensure correct column names (case-sensitive)
- Ensure CSV uses comma delimiter
- Verify data is numeric
- Download template and use it as reference

### Issue: Model not found
**Solution:**
- On first run, model will be automatically trained
- Wait for "Model trained and saved successfully" message
- If still issues, run: `python main.py` first

### Issue: matplotlib display error
**Solution:**
- This is expected in non-GUI environments
- The CLI version uses matplotlib.use('Agg') internally
- Web interface doesn't display plots (API only)

---

## Environment Variables (Optional)

Create a `.env` file to customize:

```
FLASK_ENV=development
FLASK_DEBUG=True
FLASK_PORT=5000
MODEL_PATH=models/model.pkl
```

---

## Advanced Usage

### Load Custom Dataset
```bash
# Place CSV in data/ folder with same column format
# The system will automatically use it
```

### Retrain Model
```bash
# Delete or rename models/model.pkl and models/scaler.pkl
# Re-run the application
python app.py  # or python main.py
```

### Production Deployment
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

---

## System Requirements

| Component | Requirement |
|-----------|-------------|
| **OS** | Windows 7+ / Mac / Linux |
| **Python** | 3.8+ |
| **RAM** | 512MB minimum |
| **Disk Space** | 500MB |
| **Browser** | Any modern browser |
| **Port** | 5000 (changeable) |

---

## Getting Help

### Documentation Files
- `README.md` - Original project documentation
- `README_UPDATED.md` - Comprehensive guide with all details
- `config.py` - Configuration settings

### Common Questions

**Q: Where is my data stored?**
A: In `data/student_data.csv` - CSV format for easy access

**Q: Can I use my own dataset?**
A: Yes, place CSV in `data/` folder with required columns

**Q: How accurate is the model?**
A: 90% accuracy with 92.31% precision/recall - see Dashboard for details

**Q: Can I modify the model?**
A: Edit `src/model_training.py` to adjust parameters

---

## Next Steps

1. ✓ Start the web application: `python run.py`
2. ✓ Explore the dashboard
3. ✓ Try single student predictions
4. ✓ Try batch upload with CSV
5. ✓ View model statistics

---

**Enjoy using the Student Performance Prediction System! 🎓**
