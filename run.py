"""
Run student performance prediction web application
Convenient script to start the Flask web server
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import app
from config import FLASK_HOST, FLASK_PORT, FLASK_DEBUG

if __name__ == '__main__':
    print("\n" + "="*60)
    print("STUDENT PERFORMANCE PREDICTION SYSTEM")
    print("Starting Flask Web Application")
    print("="*60)
    print(f"\n✓ Server starting on http://{FLASK_HOST}:{FLASK_PORT}")
    print("✓ Press Ctrl+C to stop the server\n")
    
    try:
        app.run(
            host=FLASK_HOST,
            port=FLASK_PORT,
            debug=FLASK_DEBUG,
            use_reloader=False
        )
    except KeyboardInterrupt:
        print("\n\n✓ Server stopped.")
        sys.exit(0)
