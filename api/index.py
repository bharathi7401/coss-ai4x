import sys
import os

# Add the parent directory to the Python path so we can import from the backend
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai4x_demo_app import app

# Vercel expects the FastAPI app to be available as a variable named 'app'
# This file serves as the entry point for Vercel deployment