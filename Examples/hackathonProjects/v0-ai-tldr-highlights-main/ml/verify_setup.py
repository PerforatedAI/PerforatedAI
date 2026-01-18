#!/usr/bin/env python3
"""
ML Setup Verification Script

Run this script to verify that all ML components are properly configured:
- PyTorch and CUDA
- HuggingFace Transformers
- Weights & Biases
- PerforatedAI (optional)

Usage:
    python -m ml.verify_setup
    
Environment Variables:
    WANDB_API_KEY: Your W&B API key
    HF_TOKEN: Your HuggingFace token (optional, for gated models)
"""

import os
import sys
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def print_section(title: str):
    logger.info("")
    logger.info("=" * 60)
    logger.info(f"  {title}")
    logger.info("=" * 60)

def check_pytorch():
    """Verify PyTorch installation."""
    print_section("PyTorch")
    
    try:
        import torch
        logger.info(f"[OK] PyTorch version: {torch.__version__}")
        logger.info(f"[OK] CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            logger.info(f"[OK] CUDA version: {torch.version.cuda}")
            logger.info(f"[OK] GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"[OK] GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            logger.warning("[WARN] CUDA not available - training will be slow on CPU")
        
        # Quick tensor test
        x = torch.randn(100, 100)
        y = torch.matmul(x, x.T)
        logger.info(f"[OK] Tensor operations working")
        
        return True
    except Exception as e:
        logger.error(f"[FAIL] PyTorch error: {e}")
        return False

def check_transformers():
    """Verify HuggingFace Transformers."""
    print_section("HuggingFace Transformers")
    
    try:
        import transformers
        logger.info(f"[OK] Transformers version: {transformers.__version__}")
        
        # Check HF token
        hf_token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGINGFACE_HUB_TOKEN')
        if hf_token:
            logger.info(f"[OK] HF_TOKEN configured (length: {len(hf_token)})")
        else:
            logger.warning("[WARN] HF_TOKEN not set - cannot access gated models")
        
        # Load a small tokenizer
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("t5-small")
        logger.info(f"[OK] Tokenizer loading works")
        
        # Tokenize test
        tokens = tokenizer("Hello, world!", return_tensors="pt")
        logger.info(f"[OK] Tokenization working: {tokens['input_ids'].shape}")
        
        return True
    except Exception as e:
        logger.error(f"[FAIL] Transformers error: {e}")
        return False

def check_wandb():
    """Verify Weights & Biases."""
    print_section("Weights & Biases")
    
    try:
        import wandb
        logger.info(f"[OK] W&B version: {wandb.__version__}")
        
        # Check API key
        api_key = os.environ.get('WANDB_API_KEY')
        if api_key:
            logger.info(f"[OK] WANDB_API_KEY configured (length: {len(api_key)})")
            
            # Test login
            try:
                wandb.login(key=api_key, relogin=True)
                logger.info(f"[OK] W&B login successful")
                
                # Quick init test (offline mode)
                run = wandb.init(
                    project="tldr-verify",
                    mode="offline",
                    config={"test": True},
                )
                wandb.log({"test_metric": 1.0})
                run.finish()
                logger.info(f"[OK] W&B logging working (offline test)")
                
            except Exception as e:
                logger.warning(f"[WARN] W&B login test failed: {e}")
        else:
            logger.error("[FAIL] WANDB_API_KEY not set")
            logger.info("       Set it with: export WANDB_API_KEY=your_key")
            return False
        
        return True
    except ImportError:
        logger.error("[FAIL] wandb not installed")
        logger.info("       Install with: pip install wandb")
        return False
    except Exception as e:
        logger.error(f"[FAIL] W&B error: {e}")
        return False

def check_perforatedai():
    """Verify PerforatedAI (optional)."""
    print_section("PerforatedAI (Optional)")
    
    try:
        import perforatedai as PA
        logger.info(f"[OK] PerforatedAI installed")
        
        # Check globals
        from perforatedai import pb_globals as PBG
        logger.info(f"[OK] PAI globals accessible")
        
        # Check converter
        import torch.nn as nn
        
        class SimpleNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(10, 20)
                self.fc2 = nn.Linear(20, 10)
            
            def forward(self, x):
                return self.fc2(torch.relu(self.fc1(x)))
        
        model = SimpleNet()
        
        # Test conversion
        try:
            PA.convert_network(model, module_names_to_convert=["fc1", "fc2"])
            logger.info(f"[OK] PAI conversion working")
        except Exception as e:
            logger.warning(f"[WARN] PAI conversion test failed: {e}")
        
        # Test tracker
        try:
            tracker = PA.PerforatedBackPropagationTracker(
                do_pb=True,
                save_name="verify_test",
                maximizing_score=True,
            )
            logger.info(f"[OK] PAI tracker initialization working")
        except Exception as e:
            logger.warning(f"[WARN] PAI tracker test failed: {e}")
        
        return True
        
    except ImportError:
        logger.warning("[WARN] PerforatedAI not installed")
        logger.info("       Install with: pip install perforatedai")
        logger.info("       This is optional - training will work without it")
        return False
    except Exception as e:
        logger.warning(f"[WARN] PerforatedAI error: {e}")
        return False

def check_datasets():
    """Verify datasets library."""
    print_section("Datasets")
    
    try:
        import datasets
        logger.info(f"[OK] Datasets version: {datasets.__version__}")
        
        # Load small sample
        from datasets import load_dataset
        ds = load_dataset("samsum", split="train[:10]", trust_remote_code=True)
        logger.info(f"[OK] SAMSum dataset loading works: {len(ds)} examples")
        
        return True
    except Exception as e:
        logger.error(f"[FAIL] Datasets error: {e}")
        return False

def check_evaluation():
    """Verify evaluation metrics."""
    print_section("Evaluation Metrics")
    
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        scores = scorer.score("hello world", "hello there")
        logger.info(f"[OK] ROUGE scoring working")
        
        return True
    except ImportError:
        logger.warning("[WARN] rouge-score not installed")
        logger.info("       Install with: pip install rouge-score")
        return False
    except Exception as e:
        logger.error(f"[FAIL] ROUGE error: {e}")
        return False

def run_mini_training():
    """Run a minimal training loop to verify everything works together."""
    print_section("Mini Training Test")
    
    try:
        import torch
        import torch.nn as nn
        from transformers import T5ForConditionalGeneration, T5Tokenizer
        
        logger.info("Loading T5-small model...")
        tokenizer = T5Tokenizer.from_pretrained("t5-small")
        model = T5ForConditionalGeneration.from_pretrained("t5-small")
        
        # Move to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        logger.info(f"[OK] Model loaded on {device}")
        
        # Create dummy batch
        input_text = "summarize: The quick brown fox jumps over the lazy dog."
        target_text = "A fox jumps over a dog."
        
        inputs = tokenizer(input_text, return_tensors="pt", max_length=64, truncation=True, padding="max_length")
        targets = tokenizer(target_text, return_tensors="pt", max_length=32, truncation=True, padding="max_length")
        
        inputs = {k: v.to(device) for k, v in inputs.items()}
        targets = {k: v.to(device) for k, v in targets.items()}
        
        # Forward pass
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=targets["input_ids"],
        )
        
        loss = outputs.loss
        logger.info(f"[OK] Forward pass working, loss: {loss.item():.4f}")
        
        # Backward pass
        loss.backward()
        logger.info(f"[OK] Backward pass working")
        
        # Generation test
        generated = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=32,
        )
        decoded = tokenizer.decode(generated[0], skip_special_tokens=True)
        logger.info(f"[OK] Generation working: '{decoded}'")
        
        return True
        
    except Exception as e:
        logger.error(f"[FAIL] Mini training error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all verification checks."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("  AI TL;DR ML Setup Verification")
    logger.info("=" * 60)
    
    results = {}
    
    results["pytorch"] = check_pytorch()
    results["transformers"] = check_transformers()
    results["wandb"] = check_wandb()
    results["perforatedai"] = check_perforatedai()
    results["datasets"] = check_datasets()
    results["evaluation"] = check_evaluation()
    results["mini_training"] = run_mini_training()
    
    # Summary
    print_section("Summary")
    
    all_passed = True
    required = ["pytorch", "transformers", "wandb", "datasets", "mini_training"]
    optional = ["perforatedai", "evaluation"]
    
    for name in required:
        status = "[OK]" if results.get(name) else "[FAIL]"
        logger.info(f"{status} {name} (required)")
        if not results.get(name):
            all_passed = False
    
    for name in optional:
        status = "[OK]" if results.get(name) else "[SKIP]"
        logger.info(f"{status} {name} (optional)")
    
    logger.info("")
    if all_passed:
        logger.info("[SUCCESS] All required components are working!")
        logger.info("")
        logger.info("Next steps:")
        logger.info("  1. Run training: python -m ml.train --help")
        logger.info("  2. Start sweep: bash scripts/launch_pai_sweep.sh")
        logger.info("  3. View results: https://wandb.ai/your-entity/tldr-highlights")
        return 0
    else:
        logger.error("[FAILURE] Some required components failed!")
        logger.info("")
        logger.info("Please fix the issues above and re-run this script.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
