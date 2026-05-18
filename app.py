import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
import re
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
import io
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# ✅ FIX: Scaler was trained on exactly 3 features — do NOT add more.
FEATURE_COLUMNS = [
    'Attendance',
    'Internal_Marks',
    'Previous_Grade_Value',
]

# Columns included in every export (CSV / Excel / PDF).
# Columns NOT listed here are internal-only and will be stripped at export time.
EXPORT_COLUMNS = [
    'Student_ID',
    'Student_Name',
    'Attendance',
    'Attendance_Marks',
    'Internal_Marks',
    'Previous_Results',
    'Previous_Grade',
    'Predicted_Performance',
    'Good_Performance_Probability',
    'Average_Performance_Probability',
    'Poor_Performance_Probability',
]

COLUMN_ALIASES = {
    'student_id': 'Student_ID',
    'id': 'Student_ID',
    'student_name': 'Student_Name',
    'name': 'Student_Name',
    'total_days': 'Total_Days',
    'days_present': 'Days_Present',
    'attendance_marks': 'Attendance_Marks',
    'attendance': 'Attendance',
    'internal_marks': 'Internal_Marks',
    'assignment_marks': 'Assignment_Marks',
    'study_hours': 'Study_Hours',
    'previous_results': 'Previous_Results',
    'previous_grade': 'Previous_Grade',
    'pass_fail': 'Pass_Fail',
    'previous_grade_value': 'Previous_Grade_Value',
    'previous_year_result': 'Previous_Results',
    'performance_label': 'Performance_Label',
    'next_exam_marks': 'Next_Exam_Marks'
}

# ✅ FIX: Map the actual grades from the CSV (O, A+, A, B+, B, C, F)
GRADE_VALUE_MAP = {
    'o':   10.0,   # Outstanding
    'a+':   9.0,
    'a':    8.0,
    'b+':   7.0,
    'b':    6.0,
    'c':    5.0,
    'f':    0.0,
    # legacy labels kept for backward compatibility
    'distinction':   10.0,
    'first class':    8.0,
    'second class':   6.0,
    'pass':           5.0,
    'fail':           0.0,
}

PASS_FAIL_MAP = {
    'pass': 1,
    'p': 1,
    'yes': 1,
    'true': 1,
    '1': 1,
    1: 1,
    'fail': 0,
    'f': 0,
    'no': 0,
    'false': 0,
    '0': 0,
    0: 0
}


def _standardize_columns(df):
    df = df.copy()
    df.columns = [col.strip() for col in df.columns]
    rename_map = {}

    for col in df.columns:
        key = col.strip().lower().replace(' ', '_').replace('-', '_')
        key = re.sub(r'[^a-z0-9_]', '', key)
        key = key.strip('_')
        # try exact or prefix match against aliases
        mapped = None
        for alias_key, alias_val in COLUMN_ALIASES.items():
            normalized_key = key.replace('_', '')
            normalized_alias = alias_key.replace('_', '')
            if (key == alias_key or key.startswith(alias_key + '_') or key.startswith(alias_key)
                    or (alias_key.endswith('s') and key.startswith(alias_key[:-1]))
                    or normalized_key == normalized_alias
                    or normalized_key.startswith(normalized_alias + '_')
                    or normalized_key.startswith(normalized_alias)):
                mapped = alias_val
                break
        rename_map[col] = mapped if mapped is not None else col

    return df.rename(columns=rename_map)


def normalize_student_dataframe(df):
    """Normalize uploaded dataset columns and convert grade labels to numeric values."""
    df = _standardize_columns(df)
    df = df.copy()

    if 'Attendance' not in df.columns and 'Days_Present' in df.columns and 'Total_Days' in df.columns:
        df['Total_Days'] = pd.to_numeric(df['Total_Days'], errors='coerce')
        df['Days_Present'] = pd.to_numeric(df['Days_Present'], errors='coerce')
        df['Attendance'] = np.where(
            df['Total_Days'] > 0,
            (df['Days_Present'] / df['Total_Days']) * 100,
            np.nan
        )

    for col in ['Attendance', 'Attendance_Marks', 'Internal_Marks', 'Previous_Results']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    if 'Previous_Grade' in df.columns:
        df['Previous_Grade_Value'] = (
            df['Previous_Grade']
            .astype(str)
            .str.strip()
            .str.lower()
            .map(GRADE_VALUE_MAP)
        )

    if 'Previous_Grade_Value' in df.columns:
        df['Previous_Grade_Value'] = pd.to_numeric(df['Previous_Grade_Value'], errors='coerce')

    if 'Pass_Fail' in df.columns:
        # drop Pass/Fail column if present (not used as input feature)
        try:
            df.drop(columns=['Pass_Fail'], inplace=True)
        except Exception:
            pass

    if 'Attendance_Marks' not in df.columns or df['Attendance_Marks'].isnull().all():
        if 'Attendance' in df.columns:
            df['Attendance_Marks'] = np.clip((df['Attendance'] / 100.0) * 5.0, 0, 5)
        else:
            df['Attendance_Marks'] = 0.0

    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = 0.0

    numeric_cols = [c for c in df.columns if c in FEATURE_COLUMNS or c in ['Total_Days', 'Days_Present']]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean().fillna(0.0))

    if not all(col in df.columns for col in ['Attendance', 'Internal_Marks', 'Previous_Results']):
        missing = [col for col in ['Attendance', 'Internal_Marks', 'Previous_Results'] if col not in df.columns]
        raise ValueError(f'Missing required columns: {missing}')

    if df[['Attendance', 'Internal_Marks', 'Previous_Results']].isnull().any().any():
        invalid_rows = df[['Attendance', 'Internal_Marks', 'Previous_Results']][
            df[['Attendance', 'Internal_Marks', 'Previous_Results']].isnull().any(axis=1)
        ]
        raise ValueError(f'Invalid or non-numeric values found in required columns. Please check rows:\n{invalid_rows.head(5)}')

    return df


def load_uploaded_file(file):
    filename = secure_filename(file.filename)
    if filename.lower().endswith('.csv'):
        return pd.read_csv(file)
    if filename.lower().endswith(('.xls', '.xlsx')):
        return pd.read_excel(file)
    raise ValueError('Unsupported file type. Only CSV and Excel files are accepted.')


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
                if hasattr(model, 'coef_') and model.coef_.shape[1] != len(FEATURE_COLUMNS):
                    print('Saved model feature count does not match current feature set. Retraining from updated dataset...')
                    train_model()
                    return
                print("✓ Model loaded from saved files")
                print(f"✓ Scaler type: {type(scaler)}")
                # initialize prediction engine
                try:
                    prediction_engine = PredictionEngine(model, scaler, feature_names=FEATURE_COLUMNS)
                    # load regression if exists
                    if os.path.exists('models/regression.pkl'):
                        try:
                            with open('models/regression.pkl', 'rb') as rf:
                                prediction_engine.regression_model = pickle.load(rf)
                            print('✓ Regression model loaded')
                        except Exception:
                            prediction_engine.regression_model = None
                    return
                except Exception as e:
                    print(f'Warning: could not initialize prediction engine: {e}')
                    return
            except Exception as load_error:
                print(f"Error loading saved model: {load_error}")
                print("Will train new model...")
        
        train_model()
        
    except Exception as e:
        print(f"Error in initialize_model: {e}")
        train_model()

def train_model():
    """Train and save the model (classification + regression for marks)"""
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

        # Train a simple linear regression to estimate continuous marks
        try:
            from marks_regression import MarksRegressionModel
            print('Training Marks Regression model...')
            full_df = pd.read_csv(data_path)
            reg = MarksRegressionModel()
            reg.train(full_df)
            with open('models/regression.pkl', 'wb') as f:
                pickle.dump(reg, f)
            print('✓ Regression model trained and saved to models/regression.pkl')
        except Exception as re:
            print(f'Warning: could not train regression model: {re}')
            reg = None
        
        # Initialize prediction engine and attach regression model if available
        try:
            prediction_engine = PredictionEngine(model, scaler, feature_names=FEATURE_COLUMNS)
            prediction_engine.regression_model = reg
            print('✓ Prediction engine initialized')
        except Exception as pe:
            print(f'Warning: could not initialize prediction engine: {pe}')
        
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
    try:
        data = request.get_json()
        if not isinstance(data, dict):
            raise ValueError('Invalid request payload')

        student_df = pd.DataFrame([data])
        normalized_df = normalize_student_dataframe(student_df)

        with open('models/scaler.pkl', 'rb') as f:
            scaler_local = pickle.load(f)
        with open('models/model.pkl', 'rb') as f:
            model_local = pickle.load(f)

        engine = PredictionEngine(model_local, scaler_local, feature_names=FEATURE_COLUMNS)
        if os.path.exists('models/regression.pkl'):
            with open('models/regression.pkl', 'rb') as f:
                engine.regression_model = pickle.load(f)

        print('DEBUG normalized_df columns:', normalized_df.columns.tolist())
        print('DEBUG engine feature_names:', engine.feature_names)
        print('DEBUG feature subset columns:', [c for c in engine.feature_names if c in normalized_df.columns])
        print('DEBUG feature subset shape:', normalized_df[engine.feature_names].shape)

        result = engine.predict_single_student(normalized_df.iloc[0].to_dict())

        return jsonify({'status': 'success', **result})
    except Exception as e:
        debug_info = {}
        try:
            debug_info = {
                'normalized_columns': normalized_df.columns.tolist(),
                'engine_feature_names': engine.feature_names,
                'engine_feature_intersection': [c for c in engine.feature_names if c in normalized_df.columns]
            }
        except Exception:
            debug_info = {'error': 'could not collect debug info'}
        return jsonify({'status': 'error', 'message': str(e), 'debug': debug_info}), 400

@app.route('/api/batch-predict', methods=['POST'])
def api_batch_predict():
    """API endpoint for batch predictions"""
    try:
        if 'file' not in request.files:
            return jsonify({'status': 'error', 'message': 'No file provided'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'status': 'error', 'message': 'No file selected'}), 400

        df = load_uploaded_file(file)
        normalized_df = normalize_student_dataframe(df)

        with open('models/scaler.pkl', 'rb') as f:
            scaler_local = pickle.load(f)
        with open('models/model.pkl', 'rb') as f:
            model_local = pickle.load(f)

        engine = PredictionEngine(model_local, scaler_local, feature_names=FEATURE_COLUMNS)
        if os.path.exists('models/regression.pkl'):
            with open('models/regression.pkl', 'rb') as f:
                engine.regression_model = pickle.load(f)

        predictions_df = engine.predict_batch(normalized_df)
        output_columns = [
            'Student_ID', 'Student_Name',
            'Attendance', 'Attendance_Marks', 'Internal_Marks',
            'Previous_Results', 'Previous_Grade',
            'Predicted_Performance',
            'Good_Performance_Probability', 'Average_Performance_Probability', 'Poor_Performance_Probability'
        ]
        output_df = predictions_df.reindex(columns=output_columns)
        for text_col in ['Student_ID', 'Student_Name', 'Previous_Grade']:
            if text_col in output_df.columns:
                output_df[text_col] = output_df[text_col].fillna('')

        result_records = json.loads(output_df.to_json(orient='records'))

        counts = predictions_df['Predicted_Performance'].value_counts().to_dict()

        return jsonify({
            'status': 'success',
            'records': result_records,
            'total': int(len(predictions_df)),
            'good_count': int(counts.get('Good', 0)),
            'average_count': int(counts.get('Average', 0)),
            'poor_count': int(counts.get('Poor', 0))
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400


@app.route('/api/debug-predict', methods=['POST'])
def api_debug_predict():
    try:
        data = request.get_json()
        if not isinstance(data, dict):
            raise ValueError('Invalid request payload')

        student_df = pd.DataFrame([data])
        normalized_df = normalize_student_dataframe(student_df)

        with open('models/scaler.pkl', 'rb') as f:
            scaler_local = pickle.load(f)
        with open('models/model.pkl', 'rb') as f:
            model_local = pickle.load(f)

        engine = PredictionEngine(model_local, scaler_local, feature_names=FEATURE_COLUMNS)
        feature_df = engine._ensure_feature_frame(pd.DataFrame([normalized_df.iloc[0].to_dict()]))

        return jsonify({
            'status': 'success',
            'normalized_columns': normalized_df.columns.tolist(),
            'normalized_values': normalized_df.iloc[0].to_dict(),
            'engine_feature_names': engine.feature_names,
            'feature_columns': feature_df.columns.tolist(),
            'feature_shape': feature_df.shape,
            'feature_values': feature_df.iloc[0].tolist()
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

@app.route('/api/model-stats', methods=['GET'])
def api_model_stats():
    """Get model statistics"""
    try:
        stats = {
            'model_type': 'Logistic Regression',
            'features': [col.replace('_', ' ') for col in FEATURE_COLUMNS],
            'classes': ['Poor', 'Average', 'Good'],
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
                'features': [col.replace('_', ' ') for col in FEATURE_COLUMNS],
                'coefficients': [0.0] * len(FEATURE_COLUMNS),
                'intercept': 0.0,
                'note': 'Model training in progress, showing placeholder values'
            })

        coefficients = model.coef_[0]
        importance_data = {
            'features': [col.replace('_', ' ') for col in FEATURE_COLUMNS],
            'coefficients': [round(float(c), 4) for c in coefficients],
            'intercept': round(float(model.intercept_[0]), 4)
        }
        return jsonify(importance_data)
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

@app.route('/api/students', methods=['GET'])
def api_students():
    """Return students filtered by predicted performance"""
    try:
        performance = request.args.get('performance', None)
        limit = int(request.args.get('limit', 50))
        data_path = os.path.join('data', 'student_data.csv')
        if not os.path.exists(data_path):
            return jsonify({'status': 'error', 'message': 'Dataset not found'}), 400

        df = pd.read_csv(data_path)
        normalized_df = normalize_student_dataframe(df)

        engine = prediction_engine
        if engine is None:
            with open('models/model.pkl', 'rb') as f:
                m = pickle.load(f)
            with open('models/scaler.pkl', 'rb') as f:
                s = pickle.load(f)
            engine = PredictionEngine(m, s, feature_names=FEATURE_COLUMNS)
            if os.path.exists('models/regression.pkl'):
                with open('models/regression.pkl', 'rb') as rf:
                    engine.regression_model = pickle.load(rf)

        result_df = engine.predict_batch(normalized_df)
        if performance:
            key = performance.strip().title()
            if key not in ['Poor', 'Average', 'Good']:
                return jsonify({'status': 'error', 'message': 'Performance filter must be Poor, Average, or Good'}), 400
            result_df = result_df[result_df['Predicted_Performance'] == key]

        records = json.loads(result_df.head(limit).to_json(orient='records'))
        return jsonify({'status': 'success', 'records': records, 'total': int(len(result_df))})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400


@app.route('/api/top-students', methods=['GET'])
def api_top_students():
    """Return top N students by probability or predicted marks"""
    try:
        n = int(request.args.get('n', 10))
        by = request.args.get('by', 'Predicted_Marks')
        data_path = os.path.join('data', 'student_data.csv')
        if not os.path.exists(data_path):
            return jsonify({'status': 'error', 'message': 'Dataset not found'}), 400
        df = pd.read_csv(data_path)
        normalized_df = normalize_student_dataframe(df)

        pe = prediction_engine
        if pe is None:
            with open('models/model.pkl', 'rb') as f:
                m = pickle.load(f)
            with open('models/scaler.pkl', 'rb') as f:
                s = pickle.load(f)
            pe = PredictionEngine(m, s, feature_names=FEATURE_COLUMNS)
            if os.path.exists('models/regression.pkl'):
                with open('models/regression.pkl', 'rb') as f:
                    pe.regression_model = pickle.load(f)

        top = pe.get_top_students(normalized_df, n=n, by=by)
        records = json.loads(top.to_json(orient='records'))
        return jsonify({'status': 'success', 'records': records})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400


@app.route('/api/weak-students', methods=['GET'])
def api_weak_students():
    """Return bottom N (weak) students"""
    try:
        n = int(request.args.get('n', 10))
        data_path = os.path.join('data', 'student_data.csv')
        if not os.path.exists(data_path):
            return jsonify({'status': 'error', 'message': 'Dataset not found'}), 400
        df = pd.read_csv(data_path)
        normalized_df = normalize_student_dataframe(df)

        pe = prediction_engine
        if pe is None:
            with open('models/model.pkl', 'rb') as f:
                m = pickle.load(f)
            with open('models/scaler.pkl', 'rb') as f:
                s = pickle.load(f)
            pe = PredictionEngine(m, s, feature_names=FEATURE_COLUMNS)
            if os.path.exists('models/regression.pkl'):
                with open('models/regression.pkl', 'rb') as f:
                    pe.regression_model = pickle.load(f)

        weak = pe.get_bottom_students(normalized_df, n=n, by='Predicted_Marks')
        records = json.loads(weak.to_json(orient='records'))
        return jsonify({'status': 'success', 'records': records})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400


@app.route('/api/predict-marks', methods=['POST'])
def api_predict_marks():
    """Predict continuous marks using trained regression model"""
    try:
        data = request.get_json()
        if not isinstance(data, dict):
            raise ValueError('Invalid request payload')

        student_df = pd.DataFrame([data])
        normalized_df = normalize_student_dataframe(student_df)

        if not os.path.exists('models/regression.pkl'):
            return jsonify({'status': 'error', 'message': 'Regression model not available'}), 400
        with open('models/regression.pkl', 'rb') as f:
            reg = pickle.load(f)

        predicted_marks = float(reg.predict(normalized_df.iloc[[0]])[0])
        return jsonify({'status': 'success', 'predicted_marks': round(predicted_marks, 2)})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400


@app.route('/api/export-report', methods=['POST'])
def api_export_report():
    """Generate and return report in CSV, Excel, or PDF format"""
    try:
        payload = request.get_json() or {}
        fmt = payload.get('format', 'csv').lower()
        data_path = os.path.join('data', 'student_data.csv')
        if not os.path.exists(data_path):
            return jsonify({'status': 'error', 'message': 'Dataset not found'}), 400
        df = pd.read_csv(data_path)
        pe = prediction_engine
        if pe is None:
            pe = PredictionEngine(pickle.load(open('models/model.pkl','rb')), pickle.load(open('models/scaler.pkl','rb')))
        report = pe.generate_report(df)
        out_df = report['predictions_df']
        # Keep only the user-facing columns; drop internal model artefacts
        export_cols = [c for c in EXPORT_COLUMNS if c in out_df.columns]
        out_df = out_df[export_cols]

        if fmt == 'csv':
            buf = io.StringIO()
            out_df.to_csv(buf, index=False)
            buf.seek(0)
            return send_file(io.BytesIO(buf.getvalue().encode('utf-8')),
                             mimetype='text/csv',
                             as_attachment=True,
                             download_name='performance_report.csv')
        elif fmt in ('xlsx', 'excel'):
            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine='openpyxl') as writer:
                out_df.to_excel(writer, index=False, sheet_name='Report')
            buf.seek(0)
            return send_file(buf, mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                             as_attachment=True, download_name='performance_report.xlsx')
        elif fmt == 'pdf':
            # Simple PDF rendering: table saved as a matplotlib figure
            fig, ax = plt.subplots(figsize=(10, len(out_df)*0.25+2))
            ax.axis('off')
            table = ax.table(cellText=out_df.head(200).values, colLabels=out_df.columns, loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            table.scale(1, 1.2)
            buf = io.BytesIO()
            plt.tight_layout()
            fig.savefig(buf, format='pdf', bbox_inches='tight')
            plt.close(fig)
            buf.seek(0)
            return send_file(buf, mimetype='application/pdf', as_attachment=True, download_name='performance_report.pdf')
        else:
            return jsonify({'status': 'error', 'message': 'Unsupported format'}), 400

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

# --- end new endpoints ---

@app.route('/api/download-template', methods=['GET'])
def download_template():
    """Download CSV template for batch predictions"""
    try:
        template_data = {
            'Student_ID': ['STUD1001', 'STUD1002', 'STUD1003'],
            'Student_Name': ['Aarav Sharma', 'Priya Patel', 'Rohan Iyer'],
            'Total_Days': [200, 200, 200],
            'Days_Present': [180, 190, 160],
            'Attendance': [90.0, 95.0, 80.0],
            'Attendance_Marks': [4.5, 4.8, 4.0],
            'Internal_Marks': [42.0, 38.5, 25.3],
            'Assignment_Marks': [18.0, 19.5, 15.2],
            'Study_Hours': [8.0, 6.5, 4.0],
            'Previous_Results': [78.0, 85.0, 65.0],
            'Previous_Grade': ['A', 'O', 'B+'],  # ✅ FIX: use actual grade labels
            'Pass_Fail': ['Pass', 'Pass', 'Pass']
        }

        df = pd.DataFrame(template_data)
        output = io.StringIO()
        df.to_csv(output, index=False)
        output.seek(0)

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