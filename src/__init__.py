# For relative imports to work in Python 3.6
import os 
import sys

# Add project root to sys.path to allow absolute imports starting with 'src.'
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))