"""
Quick installation test for Dendritic YOLOv8 project
Run this to verify everything is set up correctly before training.
"""

import sys

def test_imports():
    """Test all required imports"""
    print("🔍 Testing imports...")

    try:
        import torch
        print(f"✅ PyTorch {torch.__version__}")

        if torch.cuda.is_available():
            print(f"✅ CUDA available - {torch.cuda.get_device_name(0)}")
        else:
            print("⚠️  CUDA not available - training will be slow")
    except ImportError as e:
        print(f"❌ PyTorch not installed: {e}")
        return False

    try:
        from ultralytics import YOLO
        print("✅ Ultralytics YOLOv8")
    except ImportError as e:
        print(f"❌ Ultralytics not installed: {e}")
        print("   Fix: pip install ultralytics")
        return False

    try:
        from perforatedai import globals_perforatedai as GPA
        from perforatedai import utils_perforatedai as UPA
        print("✅ PerforatedAI")
    except ImportError as e:
        print(f"❌ PerforatedAI not installed: {e}")
        print("   Fix: cd /path/to/PerforatedAI && pip install -e .")
        return False

    try:
        import wandb
        print("✅ Weights & Biases")
    except ImportError as e:
        print("⚠️  W&B not installed (optional): {e}")
        print("   Fix: pip install wandb")

    return True


def test_yolo_download():
    """Test YOLO model download"""
    print("\n🔍 Testing YOLOv8 model download...")

    try:
        from ultralytics import YOLO
        model = YOLO('yolov8n.pt')
        print("✅ YOLOv8n model ready")

        # Test basic inference
        import torch
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        print(f"✅ Model loaded on {device}")

        return True
    except Exception as e:
        print(f"❌ YOLOv8 download/load failed: {e}")
        return False


def test_pai_integration():
    """Test basic PAI integration"""
    print("\n🔍 Testing PerforatedAI integration...")

    try:
        import torch
        import torch.nn as nn
        from perforatedai import globals_perforatedai as GPA
        from perforatedai import utils_perforatedai as UPA

        # Create a simple test model
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(10, 20)
                self.fc2 = nn.Linear(20, 10)

            def forward(self, x):
                x = torch.relu(self.fc1(x))
                x = self.fc2(x)
                return x

        model = TestModel()

        # Test PAI initialization
        GPA.pc.set_testing_dendrite_capacity(True)
        GPA.pc.set_verbose(False)
        model = UPA.initialize_pai(model, save_name="test")

        print("✅ PAI integration working")

        # Cleanup
        import shutil
        import os
        if os.path.exists("test"):
            shutil.rmtree("test")

        return True
    except Exception as e:
        print(f"❌ PAI integration failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_gpu_memory():
    """Test GPU memory availability"""
    print("\n🔍 Testing GPU memory...")

    import torch

    if not torch.cuda.is_available():
        print("⚠️  No GPU available")
        return True

    try:
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✅ GPU Memory: {total_memory:.2f} GB")

        if total_memory < 8:
            print("⚠️  Limited GPU memory - consider reducing batch size")
            print("   Recommended: --batch 8 --imgsz 320")
        else:
            print("✅ Sufficient GPU memory for default settings")

        return True
    except Exception as e:
        print(f"❌ GPU check failed: {e}")
        return False


def print_summary():
    """Print helpful summary"""
    print("\n" + "="*60)
    print("📋 INSTALLATION TEST COMPLETE")
    print("="*60)
    print("\nNext steps:")
    print("1. Quick test (5 min):  python train_yolov8_baseline.py --epochs 5")
    print("2. Full baseline (15 min): python train_yolov8_baseline.py --epochs 50")
    print("3. Dendritic (2 hrs): python train_yolov8_dendritic.py --epochs 50")
    print("\nOr use Colab notebook for easiest experience!")
    print("="*60 + "\n")


def main():
    print("="*60)
    print("🌳 DENDRITIC YOLOV8 - INSTALLATION TEST")
    print("="*60 + "\n")

    all_tests = [
        test_imports(),
        test_yolo_download(),
        test_pai_integration(),
        test_gpu_memory(),
    ]

    if all(all_tests):
        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED - Ready to train!")
        print_summary()
        return 0
    else:
        print("\n" + "="*60)
        print("❌ SOME TESTS FAILED - Fix errors above")
        print("="*60 + "\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
