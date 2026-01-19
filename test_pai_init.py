import sys
sys.path.insert(0, 'Examples/hackathonProjects/Project-Med-Edge')

# FIX: Disable pdb breakpoints BEFORE importing PAI
import pdb
pdb.set_trace = lambda: None

import torch
import torch.nn as nn
from src.model import DermoNet_Edge
from perforatedai import globals_perforatedai as GPA
from perforatedai import utils_perforatedai as UPA
import os

os.environ["PAIEMAIL"] = "hacker_token@perforatedai.com"
os.environ["PAITOKEN"] = "MdIq5V6gSmQM+sSak1imlCJ3tzvlyfHW8cUp+4FeQN9YxLKtwtl4HQIdmgQGmsJalAyoMtWgQVQagVOe2Bjr2THpWrxqPaU9xDnvPvRMxtYn6/bOWDqsv0Hs7td5R83rG8BMVzF8neYtxiiqrWX9XEOGlfGF8NHZVzy64C7maoO3OJiM3vDrKfhpGrAWJVV6RcGZZt/qpcraH86A2erhBhMWEbLbWqp8SRPqdJxL3mQJVcKTSe3sixQ20B3rZrRMpsfsjl0aNhZBTDhGcHzba8VTEam4k2+Sb3G5T3pWk5v7gVnFu5RN0Z0lRHeHMZ+r4VqudaOlJuH10MIQWm9Uqg=="

print("=" * 60)
print("TESTING PAI INITIALIZATION")
print("=" * 60)

try:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✓ Device: {device}")
    
    model = DermoNet_Edge().to(device)
    print(f"✓ Model created: {model.__class__.__name__}")
    
    print("\nSetting PAI configuration...")
    GPA.pc.set_testing_dendrite_capacity(False)
    print("✓ set_testing_dendrite_capacity(False)")
    
    GPA.pc.set_n_epochs_to_switch(4)
    print("✓ set_n_epochs_to_switch(4)")
    
    GPA.pc.set_max_dendrites(8)
    print("✓ set_max_dendrites(8)")
    
    GPA.pc.set_verbose(True)
    print("✓ set_verbose(True)")
    
    print("\nInitializing PAI...")
    model = UPA.initialize_pai(model, save_name="TEST_PAI")
    print("✓ PAI INITIALIZED SUCCESSFULLY!")
    
    print(f"\n✓ Model type: {type(model)}")
    print(f"✓ Model device: {next(model.parameters()).device}")
    
    print("\n" + "=" * 60)
    print("SUCCESS! PAI IS WORKING!")
    print("=" * 60)
    
except Exception as e:
    print("\n" + "=" * 60)
    print("ERROR OCCURRED:")
    print("=" * 60)
    import traceback
    traceback.print_exc()
    print("=" * 60)
