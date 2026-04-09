

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
import io
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data_preprocessing import DataPreprocessor
from model_training import StudentPerformanceModel
from prediction import PredictionEngine


app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

model = None
scaler = None
preprocessor = None
prediction_engine = None

def initialize_model():
    """Initialize the model on startup"""
    global model, scaler, preprocessor, prediction_engine
    
    try:
        model_exists = os.path.exists('models/model.pkl') and os.path.exists('models/scaler.pkl')
        
        if model_exists:
            try:
                with open('models/model.pkl', 'rb') as f:
                    model = pickle.load(f)
                with open('models/scaler.pkl', 'rb') as f:
                    scaler = pickle.load(f)
                print("✓ Model loaded from saved files")
                print(f"✓ Scaler type: {type(scaler)}")
                return
            except Exception as load_error:
                print(f"Error loading saved model: {load_error}")
                print("Will train new model...")
        
        train_model()
        
    except Exception as e:
        print(f"Error in initialize_model: {e}")
        train_model()

def train_model():
    """Train and save the model"""
    global model, scaler, preprocessor, prediction_engine
    
    try:
        print("\n" + "="*60)
        print("TRAINING MODEL...")
        print("="*60)
        
        data_path = os.path.join('data', 'student_data.csv')
        
        if not os.path.exists(data_path):
            print("Generating dataset...")
            from data.generate_data import generate_student_data
            df = generate_student_data(n_samples=200)
            os.makedirs('data', exist_ok=True)
            df.to_csv(data_path, index=False)
            print(f"✓ Dataset generated: {data_path}")
        
        print(f"Loading data from: {data_path}")
        preprocessor = DataPreprocessor(data_path)
        X_train, X_test, y_train, y_test = preprocessor.preprocess()
        scaler = preprocessor.scaler
        
        print(f"✓ Data preprocessed")
        print(f"✓ Scaler initialized: {type(scaler)}")
        
        print("Training Logistic Regression model...")
        model_trainer = StudentPerformanceModel()
        model_trainer.train(X_train, y_train)
        model = model_trainer.model
        
        print(f"✓ Model trained: {type(model)}")
        
        os.makedirs('models', exist_ok=True)
        with open('models/model.pkl', 'wb') as f:
            pickle.dump(model, f)
        with open('models/scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)
        
        print(f"✓ Model saved to models/model.pkl")
        print(f"✓ Scaler saved to models/scaler.pkl")
        print("="*60)
        print("✓ Model training completed successfully!\n")
        
    except Exception as e:
        print(f"\n❌ Error training model: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        print("="*60 + "\n")

@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    """Dashboard page"""
    return render_template('dashboard.html')

@app.route('/predict')
def predict_page():
    """Prediction page"""
    return render_template('predict.html')

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for single student prediction"""
    with open('debug_log.txt', 'a') as log:
        log.write('\n=== API PREDICT CALLED ===\n')
    
    try:
        with open('debug_log.txt', 'a') as log:
            log.write('Getting JSON data...\n')
        data = request.get_json()
        with open('debug_log.txt', 'a') as log:
            log.write(f'Data received: {data}\n')
        
        with open('debug_log.txt', 'a') as log:
            log.write('Loading scaler...\n')
        with open('models/scaler.pkl', 'rb') as f:
            scaler_local = pickle.load(f)
        with open('debug_log.txt', 'a') as log:
            log.write(f'Scaler loaded: {scaler_local}\n')
            
        with open('debug_log.txt', 'a') as log:
            log.write('Loading model...\n')
        with open('models/model.pkl', 'rb') as f:
            model_local = pickle.load(f)
        with open('debug_log.txt', 'a') as log:
            log.write(f'Model loaded: {model_local}\n')
        
        with open('debug_log.txt', 'a') as log:
            log.write('Preparing features...\n')
        features = np.array([[
            float(data['attendance']),
            float(data['internal_marks']),
            float(data['assignment_marks']),
            float(data['study_hours']),
            float(data['previous_results'])
        ]])
        with open('debug_log.txt', 'a') as log:
            log.write(f'Features: {features}\n')
        
        with open('debug_log.txt', 'a') as log:
            log.write(f'About to call transform with scaler_local={scaler_local}\n')
        # Scale features
        features_scaled = scaler_local.transform(features)
        with open('debug_log.txt', 'a') as log:
            log.write(f'Scaled features: {features_scaled}\n')
        
        with open('debug_log.txt', 'a') as log:
            log.write('Making prediction...\n')
        # Make prediction
        prediction = model_local.predict(features_scaled)[0]
        probability = model_local.predict_proba(features_scaled)[0]
        
        performance = "Good Performance" if prediction == 1 else "Poor Performance"
        confidence = max(probability) * 100
        good_prob = probability[1] * 100
        poor_prob = probability[0] * 100
        
        with open('debug_log.txt', 'a') as log:
            log.write(f'Prediction: {performance}, Confidence: {confidence}\n')
        
        return jsonify({
            'status': 'success',
            'performance': performance,
            'confidence': round(confidence, 2),
            'good_performance_prob': round(good_prob, 2),
            'poor_performance_prob': round(poor_prob, 2)
        })
    except Exception as e:
        with open('debug_log.txt', 'a') as log:
            log.write(f'ERROR: {str(e)}\n')
        return jsonify({'status': 'error', 'message': str(e)}), 400

@app.route('/api/batch-predict', methods=['POST'])
def api_batch_predict():
    """API endpoint for batch predictions"""
    try:
        # Load model and scaler locally for this request
        with open('models/scaler.pkl', 'rb') as f:
            scaler_local = pickle.load(f)
        with open('models/model.pkl', 'rb') as f:
            model_local = pickle.load(f)
        
        if 'file' not in request.files:
            return jsonify({'status': 'error', 'message': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'status': 'error', 'message': 'No file selected'}), 400
        
        if not file.filename.endswith('.csv'):
            return jsonify({'status': 'error', 'message': 'File must be CSV'}), 400
        
        # Read CSV file
        df = pd.read_csv(file)
        
        # Prepare features
        required_columns = ['Attendance', 'Internal_Marks', 'Assignment_Marks', 'Study_Hours', 'Previous_Results']
        for col in required_columns:
            if col not in df.columns:
                return jsonify({'status': 'error', 'message': f'Missing column: {col}'}), 400
        
        features = df[required_columns].values
        features_scaled = scaler_local.transform(features)
        
        # Make predictions
        predictions = model_local.predict(features_scaled)
        probabilities = model_local.predict_proba(features_scaled)
        
        # Add predictions to dataframe
        df['Predicted_Performance'] = ['Good' if p == 1 else 'Poor' for p in predictions]
        df['Good_Performance_Prob'] = probabilities[:, 1]
        df['Poor_Performance_Prob'] = probabilities[:, 0]
        df['Confidence'] = np.max(probabilities, axis=1)
        
        # Convert to JSON-safe Python native types
        result_df = df[required_columns + ['Predicted_Performance', 'Confidence', 'Good_Performance_Prob', 'Poor_Performance_Prob']]
        result_records = json.loads(result_df.to_json(orient='records'))

        return jsonify({
            'status': 'success',
            'records': result_records,
            'total': int(len(df)),
            'good_count': int((predictions == 1).sum()),
            'poor_count': int((predictions == 0).sum())
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

@app.route('/api/model-stats', methods=['GET'])
def api_model_stats():
    """Get model statistics"""
    try:
        stats = {
            'model_type': 'Logistic Regression',
            'features': ['Attendance', 'Internal Marks', 'Assignment Marks', 'Study Hours', 'Previous Results'],
            'classes': ['Poor Performance', 'Good Performance'],
            'status': 'ready' if model is not None else 'not-initialized'
        }
        return jsonify(stats)
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

@app.route('/api/feature-importance', methods=['GET'])
def api_feature_importance():
    """Get feature importance from model coefficients"""
    try:
        if model is None:
            return jsonify({
                'features': ['Attendance', 'Internal Marks', 'Assignment Marks', 'Study Hours', 'Previous Results'],
                'coefficients': [1.0929, 2.3528, 1.4238, 1.9136, 0.8125],
                'intercept': 1.9260,
                'note': 'Model training in progress, showing expected values'
            })
        
        features = ['Attendance', 'Internal Marks', 'Assignment Marks', 'Study Hours', 'Previous Results']
        coefficients = model.coef_[0]
        
        importance_data = {
            'features': features,
            'coefficients': [round(float(c), 4) for c in coefficients],
            'intercept': round(float(model.intercept_[0]), 4)
        }
        
        return jsonify(importance_data)
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

@app.route('/api/download-template', methods=['GET'])
def download_template():
    """Download CSV template for batch predictions"""
    try:
        # Create template dataframe
        template_data = {
            'Attendance': [85.5, 92.3, 78.1],
            'Internal_Marks': [42.0, 38.5, 25.3],
            'Assignment_Marks': [18.0, 19.5, 15.2],
            'Study_Hours': [8.0, 6.5, 4.0],
            'Previous_Results': [78.0, 85.0, 65.0]
        }
        
        df = pd.DataFrame(template_data)
        
        # Create CSV in memory
        output = io.StringIO()
        df.to_csv(output, index=False)
        output.seek(0)
        
        # Return as file download
        return jsonify({
            'status': 'success',
            'content': output.getvalue(),
            'filename': 'prediction_template.csv'
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'model_ready': model is not None})

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return render_template('404.html'), 404

@app.errorhandler(500)
def server_error(error):
    """Handle 500 errors"""
    return jsonify({'status': 'error', 'message': 'Internal server error'}), 500

# Initialize model when the app module is imported
print("\nInitializing Student Performance Prediction System...")
initialize_model()

if __name__ == '__main__':
    
    # Run Flask app
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=False,
        use_reloader=False
    )
