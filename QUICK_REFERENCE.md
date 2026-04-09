# 🚀 Quick Reference Guide

## Start Web Application (Recommended)
```bash
python run.py
# Then open: http://localhost:5000
```

## Start CLI Application
```bash
python main.py
```

## Run Setup
```bash
python setup.py
```

---

## Key URLs

| URL | Purpose |
|-----|---------|
| `http://localhost:5000/` | Home page & overview |
| `http://localhost:5000/predict` | Single/Batch predictions |
| `http://localhost:5000/dashboard` | Model statistics |
| `http://localhost:5000/health` | Health check |

---

## API Endpoints

### Single Prediction
```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"attendance":85,"internal_marks":42,"assignment_marks":18,"study_hours":8,"previous_results":78}'
```

### Feature Importance
```bash
curl http://localhost:5000/api/feature-importance
```

### Model Stats
```bash
curl http://localhost:5000/api/model-stats
```

---

## CSV Format for Batch Upload

```csv
Attendance,Internal_Marks,Assignment_Marks,Study_Hours,Previous_Results
85.5,42.0,18.0,8.0,78.0
92.3,38.5,19.5,6.5,85.0
```

---

## Project Files

### Documentation
- `README_UPDATED.md` - Complete documentation
- `SETUP.md` - Installation & setup guide
- `PROJECT_COMPLETE.md` - Completion summary
- `QUICK_REFERENCE.md` - This file

### Application Files
- `app.py` - Web application
- `main.py` - CLI application
- `run.py` - Run script
- `config.py` - Configuration

### ML Components
- `src/data_preprocessing.py` - Data processing
- `src/model_training.py` - Model training
- `src/prediction.py` - Predictions
- `src/visualization.py` - Visualizations

### Web Interface
- `templates/*.html` - Web pages (5 files)
- `static/style.css` - Styling
- `static/script.js` - JavaScript

### Data
- `data/generate_data.py` - Dataset generator
- `data/student_data.csv` - Generated data
- `models/` - Saved ML models
- `output/` - Generated visualizations

---

## Features

✅ Single student prediction
✅ Batch CSV predictions
✅ Model dashboard
✅ Feature importance analysis
✅ REST API endpoints
✅ Responsive web design
✅ 90% model accuracy

---

## Troubleshooting

**Port in use:**
```bash
# Change port in run.py: app.run(port=5001)
```

**Import errors:**
```bash
.venv\Scripts\activate
pip install -r requirements.txt
```

**Model not found:**
- Run application once to train model
- Or run: `python main.py`

---

## Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | 90% |
| Precision | 92.31% |
| Recall | 92.31% |
| F1-Score | 92.31% |

---

## Features Analyzed

1. Attendance (0-100%)
2. Internal Marks (0-50)
3. Assignment Marks (0-20)
4. Study Hours (0-10)
5. Previous Results (0-100)

---

## Production Deployment

```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

---

**System Ready for Use! 🎉**
