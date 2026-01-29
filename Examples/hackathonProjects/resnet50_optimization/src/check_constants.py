
import sys
import os

# Add the library root to python path so we can import perforatedai
# Current file is in src/check_constants.py
# We need to go up 4 levels to get to PerforatedAI root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../')))

try:
    from perforatedai import globals_perforatedai as GPA
    print("Available constants in GPA.pc:")
    # Check attributes of GPA.pc that start with DOING_
    
    # GPA.pc is likely an instance. Let's check dir(GPA.pc)
    for attr in dir(GPA.pc):
        if attr.startswith("DOING_"):
            print(f"{attr}: {getattr(GPA.pc, attr)}")
            
except ImportError as e:
    print(f"ImportError: {e}")
    # Print sys.path to help debug
    print("sys.path:", sys.path)
except Exception as e:
    print(f"Error: {e}")
