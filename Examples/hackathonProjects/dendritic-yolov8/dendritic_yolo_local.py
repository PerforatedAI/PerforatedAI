#!/usr/bin/env python3
"""
Dendritic YOLOv8 - Local Execution Script
Run this locally if Colab kernel issues persist
"""

import os
import sys
import json
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Set W&B API Key
os.environ["WANDB_API_KEY"] = "21942b7ed5b0ebedb98e928635acff2e972a99fc"

def check_dependencies():
    """Check and install required packages"""
    required = [
        "torch", "torchvision", "ultralytics", 
        "wandb", "matplotlib", "pandas", "seaborn"
    ]
    
    missing = []
    for package in required:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"Missing packages: {missing}")
        print("Install with: pip install " + " ".join(missing))
        return False
    return True

def setup_environment():
    """Setup GPU environment and imports"""
    print("🔧 Setting up environment...")
    
    # Import after dependency check
    import wandb
    from ultralytics import YOLO
    
    # GPU setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Try PerforatedAI import
    try:
        from perforatedai import globals_perforatedai as GPA
        from perforatedai import utils_perforatedai as UPA
        perforated_available = True
        print("✅ PerforatedAI available")
    except ImportError:
        print("⚠️ PerforatedAI not available - using baseline mode")
        perforated_available = False
        
        # Create dummy classes
        class DummyGPA:
            class pc:
                @staticmethod
                def set_testing_dendrite_capacity(val): pass
                @staticmethod
                def set_verbose(val): pass
                @staticmethod
                def set_dendrite_update_mode(val): pass
            class pai_tracker:
                @staticmethod
                def set_optimizer(opt): pass
                @staticmethod
                def set_scheduler(sched): pass
                @staticmethod
                def setup_optimizer(model, opt_args, sched_args): 
                    import torch.optim as optim
                    return optim.Adam(model.parameters(), **opt_args), None
                @staticmethod
                def add_validation_score(score, model):
                    return model, False, True
        
        class DummyUPA:
            @staticmethod
            def initialize_pai(model, **kwargs):
                return model
        
        GPA = DummyGPA()
        UPA = DummyUPA()
    
    return device, wandb, YOLO, GPA, UPA, perforated_available

def count_parameters(model):
    """Count model parameters"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def measure_inference_speed(model, device, img_size=640, runs=50):
    """Measure inference speed"""
    model.eval()
    dummy_input = torch.randn(1, 3, img_size, img_size).to(device)
    
    # Warmup
    for _ in range(5):
        with torch.no_grad():
            _ = model(dummy_input)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    start = time.perf_counter()
    for _ in range(runs):
        with torch.no_grad():
            _ = model(dummy_input)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    return (time.perf_counter() - start) / runs * 1000

def run_experiment():
    """Main experiment function"""
    print("🚀 Starting Dendritic YOLOv8 Experiment")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        print("❌ Please install missing dependencies first")
        return False
    
    # Setup environment
    device, wandb, YOLO, GPA, UPA, perforated_available = setup_environment()
    
    # Initialize W&B
    wandb.init(
        project="Dendritic-YOLOv8-Local",
        name=f"local-run-{int(time.time())}",
        tags=["local", "yolov8n"],
        config={"device": device, "perforated_ai": perforated_available}
    )
    
    try:
        # Load baseline model
        print("\n🚀 Loading baseline YOLOv8n...")
        baseline_model = YOLO("yolov8n.pt")
        baseline_model.model = baseline_model.model.to(device)
        
        total_params, _ = count_parameters(baseline_model.model)
        print(f"📊 Baseline parameters: {total_params/1e6:.2f}M")
        
        # Quick baseline training
        print("\n🚀 Training baseline (3 epochs)...")
        baseline_results = baseline_model.train(
            data="coco128.yaml",
            epochs=3,
            imgsz=640,
            batch=8,
            device=device,
            project="runs/local",
            name="baseline",
            exist_ok=True,
            verbose=False
        )
        
        # Baseline validation
        print("📊 Validating baseline...")
        baseline_val = baseline_model.val(data="coco128.yaml", device=device, verbose=False)
        baseline_map50 = float(baseline_val.box.map50) if baseline_val.box.map50 else 0.0
        baseline_speed = measure_inference_speed(baseline_model.model, device)
        
        baseline_metrics = {
            "mAP50": baseline_map50,
            "params_M": total_params/1e6,
            "speed_ms": baseline_speed
        }
        
        print(f"   mAP50: {baseline_map50:.4f}")
        print(f"   Speed: {baseline_speed:.2f}ms")
        
        # Create dendritic model
        print("\n🧠 Creating dendritic model...")
        dendritic_yolo = YOLO("yolov8n.pt")
        dendritic_model = dendritic_yolo.model.to(device)
        
        if perforated_available:
            print("   Applying PerforatedAI optimization...")
            GPA.pc.set_testing_dendrite_capacity(False)
            GPA.pc.set_verbose(True)
            GPA.pc.set_dendrite_update_mode(True)
            
            try:
                dendritic_model = UPA.initialize_pai(
                    dendritic_model,
                    doing_pai=True,
                    save_name="LocalDendriticYOLO",
                    maximizing_score=True
                )
                print("✅ PerforatedAI optimization applied")
            except Exception as e:
                print(f"⚠️ PerforatedAI failed: {e}")
        
        dendritic_yolo.model = dendritic_model
        dendritic_params, _ = count_parameters(dendritic_model)
        print(f"📊 Dendritic parameters: {dendritic_params/1e6:.2f}M")
        
        # Dendritic training
        print("\n🚀 Training dendritic model (3 epochs)...")
        dendritic_results = dendritic_yolo.train(
            data="coco128.yaml",
            epochs=3,
            imgsz=640,
            batch=8,
            device=device,
            project="runs/local",
            name="dendritic",
            exist_ok=True,
            verbose=False
        )
        
        # Dendritic validation
        print("📊 Validating dendritic model...")
        dendritic_val = dendritic_yolo.val(data="coco128.yaml", device=device, verbose=False)
        dendritic_map50 = float(dendritic_val.box.map50) if dendritic_val.box.map50 else 0.0
        dendritic_speed = measure_inference_speed(dendritic_yolo.model, device)
        
        # Add validation score to PerforatedAI
        if perforated_available:
            try:
                dendritic_model, restructured, complete = GPA.pai_tracker.add_validation_score(
                    dendritic_map50, dendritic_model
                )
                if restructured:
                    print("🔄 Model restructured by PerforatedAI")
                if complete:
                    print("✅ PerforatedAI optimization complete")
            except Exception as e:
                print(f"⚠️ PerforatedAI validation failed: {e}")
        
        dendritic_metrics = {
            "mAP50": dendritic_map50,
            "params_M": dendritic_params/1e6,
            "speed_ms": dendritic_speed
        }
        
        print(f"   mAP50: {dendritic_map50:.4f}")
        print(f"   Speed: {dendritic_speed:.2f}ms")
        
        # Calculate improvements
        improvements = {
            "mAP50_change": dendritic_map50 - baseline_map50,
            "param_change_pct": ((dendritic_params - total_params) / total_params) * 100,
            "speed_change_pct": ((baseline_speed - dendritic_speed) / baseline_speed) * 100 if baseline_speed > 0 else 0
        }
        
        # Create visualization
        print("\n📊 Creating comparison chart...")
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # mAP comparison
        map_vals = [baseline_map50, dendritic_map50]
        axes[0].bar(['Baseline', 'Dendritic'], map_vals, color=['steelblue', 'coral'])
        axes[0].set_ylabel('mAP50')
        axes[0].set_title('mAP50 Comparison')
        
        # Parameters comparison
        param_vals = [total_params/1e6, dendritic_params/1e6]
        axes[1].bar(['Baseline', 'Dendritic'], param_vals, color=['steelblue', 'coral'])
        axes[1].set_ylabel('Parameters (M)')
        axes[1].set_title('Model Size')
        
        # Speed comparison
        speed_vals = [baseline_speed, dendritic_speed]
        axes[2].bar(['Baseline', 'Dendritic'], speed_vals, color=['steelblue', 'coral'])
        axes[2].set_ylabel('Inference Time (ms)')
        axes[2].set_title('Inference Speed')
        
        plt.tight_layout()
        plt.savefig('local_comparison.png', dpi=150, bbox_inches='tight')
        print("✅ Chart saved to 'local_comparison.png'")
        
        # Save results
        results = {
            "project": "Dendritic YOLOv8 - Local Execution",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "environment": {
                "device": device,
                "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
                "pytorch_version": torch.__version__,
                "perforated_ai_available": perforated_available
            },
            "baseline": baseline_metrics,
            "dendritic": dendritic_metrics,
            "improvements": improvements,
            "success": True
        }
        
        with open('local_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        # Print summary
        print("\n" + "=" * 60)
        print("🏆 LOCAL DENDRITIC YOLOV8 RESULTS")
        print("=" * 60)
        print(f"📊 Baseline:  mAP50={baseline_map50:.4f}, Params={total_params/1e6:.2f}M, Speed={baseline_speed:.1f}ms")
        print(f"📊 Dendritic: mAP50={dendritic_map50:.4f}, Params={dendritic_params/1e6:.2f}M, Speed={dendritic_speed:.1f}ms")
        print(f"\n📈 Improvements:")
        for key, value in improvements.items():
            print(f"   {key}: {value:+.3f}{'%' if 'pct' in key else ''}")
        print(f"\n🔧 PerforatedAI: {'✅ Active' if perforated_available else '❌ Not Available'}")
        print(f"🔧 Device: {device}")
        print("=" * 60)
        
        print("\n✅ Results saved to 'local_results.json'")
        print("✅ Chart saved to 'local_comparison.png'")
        
        # Log to W&B
        wandb.log({
            **{f"baseline_{k}": v for k, v in baseline_metrics.items()},
            **{f"dendritic_{k}": v for k, v in dendritic_metrics.items()},
            **{f"improvement_{k}": v for k, v in improvements.items()}
        })
        wandb.finish()
        
        return True
        
    except Exception as e:
        print(f"❌ Experiment failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_experiment()
    if success:
        print("\n🎯 Experiment completed successfully!")
    else:
        print("\n❌ Experiment failed - check error messages above")