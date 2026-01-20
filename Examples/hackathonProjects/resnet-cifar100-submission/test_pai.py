
import sys
import os
import torch
import torch.nn as nn
import traceback

# Add repository root to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(current_dir, "../../.."))
sys.path.append(repo_root)

try:
    from perforatedai import globals_perforatedai as GPA
    from perforatedai import utils_perforatedai as UPA
    print("Import successful")
except ImportError as e:
    print(f"Import failed: {e}")
    sys.exit(1)

from torchvision.models import resnet50

# ...

def test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = resnet50().to(device)
    print("Initializing PAI...")
    try:
        model = UPA.initialize_pai(model)
        print("Initialization successful")
    except Exception as e:
        print("Caught exception:")
        with open('traceback_dump.txt', 'w', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()

if __name__ == "__main__":
    test()
