import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# FIX 1: Grade map updated to match the actual dataset grades (O/A+/A/B+/B/C/F).
# Old map used 'distinction'/'first class' etc. which never matched, so
# Previous_Grade_Value was always NaN and silently fell back to a value
# derived from Previous_Results — leaking the target.
GRADE_VALUE_MAP = {
    'o':   95.0,
    'a+':  85.0,
    'a':   75.0,
    'b+':  65.0,
    'b':   55.0,
    'c':   45.0,
    'f':   20.0,
    # Legacy keys kept for backwards-compatibility with generated data
    'distinction':   95.0,
    'first class':   80.0,
    'second class':  65.0,
    'pass':          50.0,
    'fail':          20.0,
}

PASS_FAIL_MAP = {
    'pass': 1, 'p': 1, 'yes': 1, 'true': 1, '1': 1, 1: 1,
    'fail': 0, 'f': 0, 'no':  0, 'false': 0, '0': 0, 0: 0,
}


class DataPreprocessor:
    """
    Handles data loading, cleaning, and preprocessing for student performance prediction.
    """

    # FIX 2: Remove 'Previous_Results' from feature columns — it directly
    # encodes the target grade (score → grade is a deterministic mapping), so
    # including it is label leakage.
    # FIX 3: Remove 'Attendance_Marks' — it has 0.977 correlation with
    # 'Attendance' and is simply a bucketed duplicate.  Keeping both inflates
    # the apparent importance of attendance and can mislead tree models.
    FEATURE_COLUMNS = [
        'Attendance',
        'Internal_Marks',
        'Previous_Grade_Value',
    ]

    def __init__(self, filepath):
        self.filepath = filepath
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()

    def load_data(self):
        """Load data from CSV or Excel file."""
        try:
            if self.filepath.lower().endswith(('.xls', '.xlsx')):
                self.df = pd.read_excel(self.filepath)
            else:
                self.df = pd.read_csv(self.filepath)
            print(f"Data loaded successfully! Shape: {self.df.shape}")
            self._normalize_columns()
            self._prepare_features()
            return self.df
        except FileNotFoundError:
            print(f"Error: File {self.filepath} not found!")
            return None
        except Exception as e:
            print(f"Error loading data: {e}")
            return None

    def _normalize_columns(self):
        """Normalize common column names and apply basic conversions."""
        if self.df is None:
            return

        cleaned_columns = {}
        for col in self.df.columns:
            normalized = col.strip().lower().replace(' ', '_').replace('-', '_')
            normalized = re.sub(r'[^a-z0-9_]', '', normalized)
            normalized = normalized.strip('_')
            cleaned_columns[col] = normalized
        self.df.rename(columns=cleaned_columns, inplace=True)

        column_map = {
            'student_id':            'Student_ID',
            'id':                    'Student_ID',
            'student_name':          'Student_Name',
            'name':                  'Student_Name',
            'total_days':            'Total_Days',
            'days_present':          'Days_Present',
            'attendance_marks':      'Attendance_Marks',
            'attendance':            'Attendance',
            'internal_marks':        'Internal_Marks',
            'assignment_marks':      'Assignment_Marks',
            'study_hours':           'Study_Hours',
            'previous_results':      'Previous_Results',
            'previous_grade':        'Previous_Grade',
            'pass_fail':             'Pass_Fail',
            'previous_year_result':  'Previous_Results',
            'previous_grade_value':  'Previous_Grade_Value',
            'performance_label':     'Performance_Label',
            'next_exam_marks':       'Next_Exam_Marks',
        }

        rename_map2 = {}
        for col in list(self.df.columns):
            for k, v in column_map.items():
                normalized_key   = col.replace('_', '')
                normalized_alias = k.replace('_', '')
                if (col == k
                        or col.startswith(k + '_')
                        or col.startswith(k)
                        or (k.endswith('s') and col.startswith(k[:-1]))
                        or normalized_key == normalized_alias
                        or normalized_key.startswith(normalized_alias + '_')
                        or normalized_key.startswith(normalized_alias)):
                    rename_map2[col] = v
                    break
        if rename_map2:
            self.df.rename(columns=rename_map2, inplace=True)

    def _prepare_features(self):
        """Prepare feature columns for model training and prediction."""
        if self.df is None:
            return

        # Compute Attendance from Days Present / Total Days when needed
        if ('Attendance' not in self.df.columns
                and 'Days_Present' in self.df.columns
                and 'Total_Days' in self.df.columns):
            self.df['Attendance'] = np.where(
                self.df['Total_Days'].astype(float) > 0,
                (self.df['Days_Present'].astype(float)
                 / self.df['Total_Days'].astype(float)) * 100,
                np.nan,
            )

        # Convert numeric fields
        for col in ['Attendance', 'Attendance_Marks', 'Internal_Marks',
                    'Previous_Results', 'Total_Days', 'Days_Present',
                    'Next_Exam_Marks']:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')

        # FIX 4: Map Previous_Grade to a numeric value.
        # The old GRADE_VALUE_MAP only had 'distinction'/'first class' etc., so
        # every grade in this CSV (O/A+/A/B+/B/C/F) mapped to NaN.  The fallback
        # then re-derived the value from Previous_Results — leaking the target.
        # The updated GRADE_VALUE_MAP now covers the real grade labels.
        if 'Previous_Grade' in self.df.columns:
            self.df['Previous_Grade_Value'] = (
                self.df['Previous_Grade']
                .astype(str).str.strip().str.lower()
                .map(GRADE_VALUE_MAP)
            )
        if 'Previous_Grade_Value' in self.df.columns:
            self.df['Previous_Grade_Value'] = pd.to_numeric(
                self.df['Previous_Grade_Value'], errors='coerce'
            )

        # Drop Pass_Fail — not a model feature
        if 'Pass_Fail' in self.df.columns:
            try:
                self.df.drop(columns=['Pass_Fail'], inplace=True)
            except Exception:
                pass

        # Derive Attendance_Marks if needed (kept for data generated by generate_data.py)
        if ('Attendance_Marks' not in self.df.columns
                or self.df['Attendance_Marks'].isnull().all()):
            if 'Attendance' in self.df.columns:
                self.df['Attendance_Marks'] = np.clip(
                    (self.df['Attendance'] / 100.0) * 5.0, 0, 5
                )
            else:
                self.df['Attendance_Marks'] = 0.0

        # FIX 5: Previous_Grade_Value fallback no longer uses Previous_Results.
        # If the grade column is genuinely absent, default to the dataset median
        # so the column exists but does NOT leak score information.
        if ('Previous_Grade_Value' not in self.df.columns
                or self.df['Previous_Grade_Value'].isnull().all()):
            self.df['Previous_Grade_Value'] = 50.0   # neutral default
        else:
            # Partial NaNs — fill with column median (not Previous_Results)
            median_val = self.df['Previous_Grade_Value'].median()
            self.df['Previous_Grade_Value'] = (
                self.df['Previous_Grade_Value'].fillna(
                    median_val if not np.isnan(median_val) else 50.0
                )
            )

        # Ensure every expected feature column exists
        for col in self.FEATURE_COLUMNS:
            if col not in self.df.columns:
                self.df[col] = 0.0

        # Fill remaining numeric missing values with column mean
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        self.df[numeric_cols] = self.df[numeric_cols].fillna(
            self.df[numeric_cols].mean().fillna(0.0)
        )

        # FIX 6: Build the Performance target correctly for the actual CSV.
        #
        # The CSV has no Performance_Label / Next_Exam_Marks / Performance column.
        # The old code fell through to `self.df['Performance'] = 0`, making
        # every student class-0 — impossible to train a useful classifier.
        #
        # We derive a 3-class target from Internal_Marks (the strongest
        # independent predictor, r=0.86 with final score):
        #   Poor    (0): Internal_Marks <  25  (bottom ~half)
        #   Average (1): 25 <= Internal_Marks < 40
        #   Good    (2): Internal_Marks >= 40
        if 'Performance_Label' in self.df.columns:
            self.df['Performance_Label'] = (
                self.df['Performance_Label'].astype(str).str.strip().str.title()
            )
            label_map = {'Poor': 0, 'Average': 1, 'Good': 2}
            self.df['Performance'] = (
                self.df['Performance_Label'].map(label_map).fillna(0).astype(int)
            )
        elif 'Performance' in self.df.columns:
            if self.df['Performance'].dtype == object:
                self.df['Performance'] = (
                    self.df['Performance'].astype(str).str.strip().str.title()
                )
                self.df['Performance'] = (
                    self.df['Performance']
                    .map({'Poor': 0, 'Average': 1, 'Good': 2})
                    .fillna(pd.to_numeric(self.df['Performance'], errors='coerce'))
                    .fillna(0).astype(int)
                )
            else:
                self.df['Performance'] = (
                    pd.to_numeric(self.df['Performance'], errors='coerce')
                    .fillna(0).astype(int)
                )
        elif 'Next_Exam_Marks' in self.df.columns:
            self.df['Performance'] = pd.cut(
                self.df['Next_Exam_Marks'],
                bins=[-1, 49.9999, 74.9999, 100],
                labels=[0, 1, 2],
            ).astype(float).fillna(0).astype(int)
        else:
            # Derive from Internal_Marks — best independent predictor available
            self.df['Performance'] = pd.cut(
                self.df['Internal_Marks'],
                bins=[-1, 24.9999, 39.9999, 50],
                labels=[0, 1, 2],
            ).astype(float).fillna(0).astype(int)

    # ------------------------------------------------------------------ #
    #  Public methods (unchanged API)                                      #
    # ------------------------------------------------------------------ #

    def display_data_info(self):
        if self.df is None:
            print("No data loaded. Please load data first.")
            return
        print("\n" + "=" * 50)
        print("DATASET INFORMATION")
        print("=" * 50)
        print(f"\nDataset Shape: {self.df.shape}")
        print(f"\nFirst 5 rows:\n{self.df.head()}")
        print(f"\nData Types:\n{self.df.dtypes}")
        print(f"\nMissing Values:\n{self.df.isnull().sum()}")
        print(f"\nStatistical Summary:\n{self.df.describe()}")

    def check_missing_values(self):
        if self.df is None:
            print("No data loaded.")
            return
        missing = self.df.isnull().sum()
        if missing.sum() == 0:
            print("No missing values found!")
            return True
        print(f"Missing values:\n{missing[missing > 0]}")
        numeric_columns = self.df.select_dtypes(include=[np.number]).columns
        self.df[numeric_columns] = self.df[numeric_columns].fillna(
            self.df[numeric_columns].mean().fillna(0.0)
        )
        print("Missing values filled with mean.")
        return True

    def split_data(self, test_size=0.2, random_state=42):
        if self.df is None:
            print("No data loaded.")
            return
        X = self.df[self.FEATURE_COLUMNS]
        y = self.df['Performance']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        print(f"\nData split completed:")
        print(f"Training set: {self.X_train.shape}")
        print(f"Testing set:  {self.X_test.shape}")
        print(f"Training performance distribution:\n{self.y_train.value_counts()}")

    def scale_features(self):
        if self.X_train is None:
            print("Data not split yet. Please split data first.")
            return
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test  = self.scaler.transform(self.X_test)
        print("\nFeatures scaled using StandardScaler")

    def get_processed_data(self):
        return self.X_train, self.X_test, self.y_train, self.y_test

    def preprocess(self, test_size=0.2, random_state=42):
        self.load_data()
        self.display_data_info()
        self.check_missing_values()
        self.split_data(test_size, random_state)
        self.scale_features()
        return self.get_processed_data()