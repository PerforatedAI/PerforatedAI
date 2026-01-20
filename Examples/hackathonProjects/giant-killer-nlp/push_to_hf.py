"""
Script to push the trained Dendritic BERT-Tiny model to Hugging Face Hub

This script:
1. Loads the trained model checkpoint
2. Prepares model card and metadata
3. Pushes to Hugging Face Hub with proper configuration
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from huggingface_hub import HfApi, create_repo
import argparse
import shutil


def load_trained_model(checkpoint_path, model_name="prajjwal1/bert-tiny", num_labels=2):
    """Load the trained model from checkpoint."""
    print(f"Loading model from {checkpoint_path}...")
    
    # Load the base model configuration
    config = AutoConfig.from_pretrained(model_name)
    config.num_labels = num_labels
    
    # Create model structure (don't load pretrained weights yet)
    model = AutoModelForSequenceClassification.from_config(config)
    
    # Load trained weights from checkpoint with weights_only=False to avoid security check
    # This is safe because we trust our own checkpoint file
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    except TypeError:
        # Older PyTorch versions don't have weights_only parameter
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Extract model state dict (handle different checkpoint formats)
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # Load state dict
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    
    if missing_keys:
        print(f"⚠ Missing keys: {missing_keys[:5]}..." if len(missing_keys) > 5 else f"⚠ Missing keys: {missing_keys}")
    if unexpected_keys:
        print(f"⚠ Unexpected keys: {unexpected_keys[:5]}..." if len(unexpected_keys) > 5 else f"⚠ Unexpected keys: {unexpected_keys}")
    
    print("✓ Model loaded successfully")
    
    return model


def prepare_model_files(model, tokenizer, output_dir="./hf_model"):
    """Save model and tokenizer in HuggingFace format."""
    print(f"\nPreparing model files in {output_dir}...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model
    model.save_pretrained(output_dir)
    print("✓ Model saved")
    
    # Save tokenizer
    tokenizer.save_pretrained(output_dir)
    print("✓ Tokenizer saved")
    
    # Copy model card
    if os.path.exists("MODEL_CARD.md"):
        shutil.copy("MODEL_CARD.md", os.path.join(output_dir, "README.md"))
        print("✓ Model card copied")
    
    # Copy config and other metadata files
    files_to_copy = [
        "configs/config.yaml",
        "requirements.txt",
        "TRAINING_SUMMARY.md",
        "THRESHOLD_OPTIMIZATION_REPORT.md"
    ]
    
    for file_path in files_to_copy:
        if os.path.exists(file_path):
            filename = os.path.basename(file_path)
            shutil.copy(file_path, os.path.join(output_dir, filename))
            print(f"✓ Copied {filename}")
    
    print("\nAll model files prepared!")
    return output_dir


def push_to_hub(model, tokenizer, repo_id, token=None, commit_message="Upload dendritic BERT-Tiny toxicity classifier"):
    """Push model directly to Hugging Face Hub."""
    print(f"\nPushing to Hugging Face Hub: {repo_id}")
    
    try:
        # Create repository if it doesn't exist
        api = HfApi()
        
        if token:
            api.token = token
        
        # Try to create repo (will fail if exists, which is fine)
        try:
            create_repo(repo_id, token=token, exist_ok=True)
            print(f"✓ Repository created/verified: {repo_id}")
        except Exception as e:
            print(f"Note: {e}")
        
        # Push model
        model.push_to_hub(
            repo_id,
            token=token,
            commit_message=commit_message,
            use_auth_token=token
        )
        print("✓ Model pushed")
        
        # Push tokenizer
        tokenizer.push_to_hub(
            repo_id,
            token=token,
            commit_message=commit_message,
            use_auth_token=token
        )
        print("✓ Tokenizer pushed")
        
        # Upload additional files
        files_to_upload = {
            "MODEL_CARD.md": "README.md",
            "configs/config.yaml": "config.yaml",
            "requirements.txt": "requirements.txt",
            "TRAINING_SUMMARY.md": "TRAINING_SUMMARY.md",
            "THRESHOLD_OPTIMIZATION_REPORT.md": "THRESHOLD_OPTIMIZATION_REPORT.md"
        }
        
        for local_path, hub_path in files_to_upload.items():
            if os.path.exists(local_path):
                try:
                    api.upload_file(
                        path_or_fileobj=local_path,
                        path_in_repo=hub_path,
                        repo_id=repo_id,
                        token=token
                    )
                    print(f"✓ Uploaded {hub_path}")
                except Exception as e:
                    print(f"⚠ Could not upload {hub_path}: {e}")
        
        print(f"\n✅ Successfully pushed to: https://huggingface.co/{repo_id}")
        
    except Exception as e:
        print(f"\n❌ Error pushing to hub: {e}")
        print("\nTroubleshooting tips:")
        print("1. Make sure you're logged in: huggingface-cli login")
        print("2. Check your token has write access")
        print("3. Verify the repo_id format: username/model-name")
        raise


def main():
    parser = argparse.ArgumentParser(description="Push trained model to Hugging Face Hub")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/best_model.pt",
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        required=True,
        help="HuggingFace repo ID (e.g., username/dendritic-bert-tiny-toxicity)"
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="HuggingFace API token (or use huggingface-cli login)"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="prajjwal1/bert-tiny",
        help="Base model name"
    )
    parser.add_argument(
        "--local_only",
        action="store_true",
        help="Only prepare files locally, don't push to hub"
    )
    
    args = parser.parse_args()
    
    # Check if checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"❌ Checkpoint not found: {args.checkpoint}")
        print("\nAvailable checkpoints:")
        if os.path.exists("checkpoints"):
            for f in os.listdir("checkpoints"):
                print(f"  - checkpoints/{f}")
        return
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    print("✓ Tokenizer loaded")
    
    # Load trained model
    model = load_trained_model(args.checkpoint, args.base_model, num_labels=2)
    
    # Set model to eval mode
    model.eval()
    
    # Print model info
    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel Info:")
    print(f"  Parameters: {num_params:,}")
    print(f"  Size: {num_params * 4 / (1024**2):.2f} MB (fp32)")
    
    if args.local_only:
        # Only prepare files locally
        output_dir = prepare_model_files(model, tokenizer)
        print(f"\n✅ Model files prepared in: {output_dir}")
        print(f"\nTo push to hub later, run:")
        print(f"  python push_to_hf.py --checkpoint {args.checkpoint} --repo_id {args.repo_id}")
    else:
        # Push directly to hub
        push_to_hub(model, tokenizer, args.repo_id, args.token)
        
        print("\n" + "="*60)
        print("🎉 SUCCESS! Your model is now on Hugging Face!")
        print("="*60)
        print(f"\n📍 Model URL: https://huggingface.co/{args.repo_id}")
        print("\nNext steps:")
        print(f"1. Visit your model page and update the README if needed")
        print(f"2. Test the model: from transformers import pipeline; classifier = pipeline('text-classification', model='{args.repo_id}')")
        print(f"3. Share your model with the community!")


if __name__ == "__main__":
    main()
