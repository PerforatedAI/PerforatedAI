import subprocess
import sys
import re
import time
import argparse

def run_benchmark(epochs=15):
    # For testing, we might want fewer splits or just one
    splits = [0.1, 0.25, 0.5, 0.75, 1.0]
    modes = ['standard', 'dendritic']
    
    results = {mode: {} for mode in modes}
    
    import os
    
    # Setup Environment with PYTHONPATH
    env = os.environ.copy()
    # Assuming the script is in .../Examples/hackathonProjects/dendritic_mobilenet
    # And we want to import from .../PerforatedAI
    # So we need to go up 3 levels from the script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pai_root = os.path.abspath(os.path.join(script_dir, '../../../'))
    
    current_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{pai_root}{os.pathsep}{current_pythonpath}"
    env["WANDB_MODE"] = "online"
    
    # Header
    print(f"Running Benchmark with {epochs} epochs/run")
    print(f"{'Mode':<12} | {'Split':<6} | {'Status':<10} | {'Result'}")
    print("-" * 45)

    for split in splits:
        for mode in modes:
            print(f"{mode:<12} | {split:<6} | Running... ", end='', flush=True)
            
            cmd = [sys.executable, 'train.py', '--split', str(split), '--mode', mode, '--epochs', str(epochs)]
            
            start_time = time.time()
            try:
                # Capture output to parse result
                # Pass the modified environment
                result = subprocess.run(
                    cmd, 
                    capture_output=True, 
                    text=True, 
                    cwd=script_dir,
                    env=env
                )
                
                output = result.stdout
                
                # Parse the final result
                match = re.search(r"FINAL_RESULT: ([\d\.]+)", output)
                if match:
                    acc = float(match.group(1))
                    results[mode][split] = acc
                    print(f"Done ({acc:.2f}%) [{int(time.time()-start_time)}s]")
                else:
                    # Check for explicit failure or missing result
                    if "FINAL_RESULT: FAILED" in output or result.returncode != 0:
                         print("Failed")
                         print(f"Error Output:\n{result.stderr}")
                         results[mode][split] = "FAILED"
                    else:
                         print("Unknown Result")
                         # print(output) # Debug if needed
                         results[mode][split] = "N/A"
                         
            except Exception as e:
                print(f"Error: {e}")
                results[mode][split] = "ERROR"

    # Final Summary Table
    print("\n\n=== Benchmark Results Summary ===")
    print(f"{'Split':<10} | {'Standard Acc':<15} | {'Dendritic Acc':<15} | {'Delta':<10}")
    print("-" * 55)
    
    for split in splits:
        std = results['standard'].get(split, 'N/A')
        den = results['dendritic'].get(split, 'N/A')
        
        std_val = std if isinstance(std, (int, float)) else 0.0
        den_val = den if isinstance(den, (int, float)) else 0.0
        
        delta = den_val - std_val
        delta_str = f"{delta:+.2f}%" if isinstance(std, float) and isinstance(den, float) else "N/A"
        
        std_str = f"{std:.2f}%" if isinstance(std, float) else str(std)
        den_str = f"{den:.2f}%" if isinstance(den, float) else str(den)
        
        print(f"{split:<10} | {std_str:<15} | {den_str:<15} | {delta_str:<10}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=15, help="Number of epochs per run")
    args = parser.parse_args()
    
    run_benchmark(epochs=args.epochs)
