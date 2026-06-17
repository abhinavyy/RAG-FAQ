import sys
import os

# Add the project root to python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.app import demo

if __name__ == "__main__":
    demo.launch()
