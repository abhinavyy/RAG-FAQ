import os
import sys

# Insert the root directory into the path so we can import from src
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Import the Gradio demo block from your src/app.py
from src.app import demo

# Launch the app
if __name__ == "__main__":
    demo.launch()
