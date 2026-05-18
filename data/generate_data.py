import pandas as pd
import numpy as np

GRADE_VALUE_MAP = {
    'Distinction': 95.0,
    'First Class': 80.0,
    'Second Class': 65.0,
    'Pass': 50.0,
    'Fail': 20.0
}

FIRST_NAMES = [
    'Aarav', 'Aditi', 'Rohan', 'Priya', 'Nisha', 'Karan', 'Parth', 'Isha', 'Sana', 'Riya',
    'Aditya', 'Ananya', 'Sai', 'Kavya', 'Riya', 'Arjun', 'Meera', 'Vikram', 'Nivedita', 'Isha'
]
LAST_NAMES = [
    'Sharma', 'Patel', 'Iyer', 'Gupta', 'Singh', 'Kumar', 'Verma', 'Kapoor', 'Mehta', 'Nair'
]

GRADE_LABELS = [
    ('Distinction', 85, 100),
    ('First Class', 70, 84),
    ('Second Class', 50, 69),
    ('Pass', 40, 49),
    ('Fail', 0, 39)
]


def get_grade_label(score):
    for label, lower, upper in GRADE_LABELS:
        if lower <= score <= upper:
            return label
    return 'Fail'


def get_pass_fail(score):
    return 'Pass' if score >= 40 else 'Fail'


def generate_student_data(n_samples=200, random_state=42):
    """
    Generate synthetic student performance dataset with expanded student fields.
    """
    np.random.seed(random_state)

    total_days = np.random.randint(180, 221, n_samples)
    attendance = np.random.uniform(50, 100, n_samples)
    days_present = np.round(total_days * (attendance / 100)).astype(int)
    attendance = np.round(np.where(total_days > 0, days_present / total_days * 100, attendance), 2)
    attendance_marks = np.round(np.clip((attendance / 100) * 5 + np.random.normal(0, 0.3, n_samples), 0, 5), 2)
    internal_marks = np.round(np.random.uniform(0, 50, n_samples), 2)
    assignment_marks = np.round(np.random.uniform(0, 20, n_samples), 2)
    study_hours = np.round(np.random.uniform(0, 10, n_samples), 2)
    previous_results = np.round(np.random.uniform(20, 100, n_samples), 2)
    previous_grades = [get_grade_label(score) for score in previous_results]
    pass_fail = [get_pass_fail(score) for score in previous_results]

    student_ids = [f'STUD{1000 + i}' for i in range(n_samples)]
    student_names = [
        f'{np.random.choice(FIRST_NAMES)} {np.random.choice(LAST_NAMES)}'
        for _ in range(n_samples)
    ]

    performance_score = (
        (attendance / 100) * 0.18 +
        (attendance_marks / 5) * 0.05 +
        (internal_marks / 50) * 0.25 +
        (assignment_marks / 20) * 0.15 +
        (study_hours / 10) * 0.15 +
        (previous_results / 100) * 0.18 +
        (np.array([1 if pf == 'Pass' else 0 for pf in pass_fail]) * 0.04)
    )
    performance_score += np.random.normal(0, 0.04, n_samples)
    performance = (performance_score >= 0.55).astype(int)

    next_exam_marks = np.round(np.clip(
        attendance * 0.18 +
        attendance_marks * 7.5 +
        internal_marks * 0.24 +
        assignment_marks * 1.5 +
        study_hours * 2.8 +
        previous_results * 0.18 +
        np.array([5 if p == 'Pass' else -10 for p in pass_fail]) +
        np.random.normal(0, 5, n_samples),
        0, 100
    ), 2)

    performance_label = [
        'Good' if mark >= 75 else 'Average' if mark >= 50 else 'Poor'
        for mark in next_exam_marks
    ]

    df = pd.DataFrame({
        'Student_ID': student_ids,
        'Student_Name': student_names,
        'Total_Days': total_days,
        'Days_Present': days_present,
        'Attendance': attendance,
        'Attendance_Marks': attendance_marks,
        'Internal_Marks': internal_marks,
        'Assignment_Marks': assignment_marks,
        'Study_Hours': study_hours,
        'Previous_Results': previous_results,
        'Previous_Grade': previous_grades,
        'Pass_Fail': pass_fail,
        'Performance': performance,
        'Performance_Label': performance_label,
        'Next_Exam_Marks': next_exam_marks
    })

    return df


if __name__ == '__main__':
    df = generate_student_data(n_samples=200)
    df.to_csv('student_data.csv', index=False)
    print('Dataset generated successfully!')
    print(f'\nDataset shape: {df.shape}')
    print(f'\nFirst few rows:\n{df.head()}')
    print(f'\nDataset info:\n{df.info()}')
    print(f'\nPerformance distribution:\n{df['Performance'].value_counts()}')