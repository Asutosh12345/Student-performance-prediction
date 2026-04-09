#!/usr/bin/env python
"""
Setup and configuration script for Student Performance Prediction System
Ensures all dependencies are installed and models are ready
"""

import os
import sys
import subprocess
import pickle

def check_python_version():
    """Check Python version"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ required")
        return False
    print(f"✓ Python {version.major}.{version.minor}.{version.micro}")
    return True

def check_virtual_environment():
    """Check if running in virtual environment"""
    in_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
    if in_venv:
        print(f"✓ Virtual environment active: {sys.prefix}")
        return True
    print("⚠ Not in virtual environment")
    return False

def install_dependencies():
    """Install required dependencies"""
    print("\n" + "="*60)
    print("INSTALLING DEPENDENCIES")
    print("="*60)
    
    try:
        requirements_file = os.path.join(os.path.dirname(__file__), 'requirements.txt')
        subprocess.check_call([
            sys.executable, '-m', 'pip', 'install', '-r', requirements_file
        ])
        print("✓ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install dependencies")
        return False

def check_data_directory():
    """Check if data directory exists and has dataset"""
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    data_file = os.path.join(data_dir, 'student_data.csv')
    if os.path.exists(data_file):
        print(f"✓ Dataset found: {data_file}")
        return True
    else:
        print("⚠ Dataset not found, will be generated on first run")
        return False

def check_models_directory():
    """Check if models directory exists"""
    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    model_file = os.path.join(models_dir, 'model.pkl')
    scaler_file = os.path.join(models_dir, 'scaler.pkl')
    
    if os.path.exists(model_file) and os.path.exists(scaler_file):
        print(f"✓ Trained model found")
        return True
    else:
        print("⚠ Trained model not found, will be trained on first run")
        return False

def check_directories():
    """Check and create necessary directories"""
    print("\n" + "="*60)
    print("CHECKING DIRECTORIES")
    print("="*60)
    
    dirs = ['data', 'models', 'templates', 'static', 'output']
    for dir_name in dirs:
        dir_path = os.path.join(os.path.dirname(__file__), dir_name)
        os.makedirs(dir_path, exist_ok=True)
        print(f"✓ Directory ready: {dir_name}/")

def print_system_info():
    """Print system information"""
    print("\n" + "="*60)
    print("SYSTEM INFORMATION")
    print("="*60)
    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {sys.platform}")
    print(f"Working directory: {os.getcwd()}")

def print_usage_info():
    """Print usage information"""
    print("\n" + "="*60)
    print("USAGE")
    print("="*60)
    print("\n1. START WEB INTERFACE:")
    print("   python app.py")
    print("   Then open: http://localhost:5000")
    
    print("\n2. RUN CLI VERSION:")
    print("   python main.py")
    
    print("\n3. GENERATE DATASET:")
    print("   python data/generate_data.py")

def main():
    """Main setup function"""
    print("\n" + "="*60)
    print("STUDENT PERFORMANCE PREDICTION SYSTEM - SETUP")
    print("="*60 + "\n")
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Check virtual environment
    check_virtual_environment()
    
    # Print system info
    print_system_info()
    
    # Check directories
    check_directories()
    
    # Check data and model directories
    print("\n" + "="*60)
    print("CHECKING PROJECT FILES")
    print("="*60)
    check_data_directory()
    check_models_directory()
    
    # Install dependencies
    if not install_dependencies():
        sys.exit(1)
    
    # Print usage information
    print_usage_info()
    
    print("\n" + "="*60)
    print("✓ SETUP COMPLETE!")
    print("="*60 + "\n")

if __name__ == '__main__':
    main()
