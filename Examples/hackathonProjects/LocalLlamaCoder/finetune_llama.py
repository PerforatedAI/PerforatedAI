"""
LocalLlama Coder - Fine-tuning Script with PerforatedAI
Integrates dendritic optimization with Transformer models (CodeLlama/Llama)
Follows PerforatedAI API requirements for proper graph generation
"""

import argparse
import yaml
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from perforatedai import globals_perforatedai as GPA
from perforatedai import utils_perforatedai as UPA
from tqdm import tqdm

from utils.pai_transformer import (
    setup_pai_lora,
    configure_transformer_tracker,
    add_validation_score_and_check_restructure,
    add_training_score,
    print_pai_info
)
from utils.code_dataset import load_code_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Llama with PerforatedAI")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--demo", action="store_true", help="Run quick demo with limited steps")
    parser.add_argument("--steps", type=int, default=50, help="Number of training steps for demo")
    parser.add_argument("--model", type=str, help="Override model name from config")
    parser.add_argument("--epochs", type=int, default=100, help="Maximum epochs (PAI will determine when to stop)")
    parser.add_argument("--batch-size", type=int, help="Override batch size")
    parser.add_argument("--load-in-8bit", action="store_true", help="Use 8-bit quantization")
    return parser.parse_args()


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def train_epoch(model, train_loader, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    num_batches = 0
    
    for batch in tqdm(train_loader, desc="Training"):
        # Move to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        loss = outputs.loss
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    avg_loss = total_loss / num_batches
    # Convert loss to score (higher is better)
    train_score = 1.0 / (1.0 + avg_loss)
    return train_score, avg_loss


def evaluate(model, eval_loader, device):
    """Evaluate the model"""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss
            
            total_loss += loss.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches
    # Convert loss to score (higher is better for PAI)
    val_score = 1.0 / (1.0 + avg_loss)
    return val_score, avg_loss


def main():
    args = parse_args()
    config = load_config(args.config)
    
    # Override config with command-line arguments
    if args.model:
        config['model']['name'] = args.model
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.load_in_8bit:
        config['model']['load_in_8bit'] = True
    
    print("🦙 LocalLlama Coder - Fine-tuning with PerforatedAI")
    print(f"Model: {config['model']['name']}")
    print(f"LoRA Rank: {config['lora']['r']}")
    print(f"Max Epochs: {args.epochs}")
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # 1. Load tokenizer and model
    print("\n📥 Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(config['model']['name'])
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        config['model']['name'],
        load_in_8bit=config['model']['load_in_8bit'],
        torch_dtype=torch.float16 if config['model']['torch_dtype'] == 'float16' else torch.float32,
        device_map=config['model']['device_map']
    )
    
    # 2. Apply LoRA
    print("\n🔧 Applying LoRA configuration...")
    lora_config = LoraConfig(
        r=config['lora']['r'],
        lora_alpha=config['lora']['lora_alpha'],
        target_modules=config['lora']['target_modules'],
        lora_dropout=config['lora']['lora_dropout'],
        bias=config['lora']['bias'],
        task_type=TaskType.CAUSAL_LM
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # 3. Initialize PerforatedAI BEFORE moving to device
    print("\n🧠 Initializing PerforatedAI on LoRA adapters...")
    model = setup_pai_lora(
        model,
        forward_function=config['pai']['forward_function'],
        correlation_threshold=config['pai']['correlation_threshold'],
        save_name="PAI_LocalLlamaCoder"
    )
    
    # Move to device AFTER PAI initialization
    if not config['model']['load_in_8bit']:
        model.to(device)
    
    # 4. Load dataset
    print("\n📚 Loading code dataset...")
    if args.demo:
        dataset_config = config['dataset'].copy()
        dataset_config['demo_mode'] = True
        dataset_config['num_samples'] = args.steps * config['training']['batch_size']
    else:
        dataset_config = config['dataset']
    
    train_dataset, eval_dataset = load_code_dataset(
        tokenizer,
        dataset_config
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Eval samples: {len(eval_dataset)}")
    
    # 5. Create data loaders
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        collate_fn=data_collator,
        shuffle=True
    )
    
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=config['training']['batch_size'],
        collate_fn=data_collator,
        shuffle=False
    )
    
    # 6. Configure PAI tracker with optimizer and scheduler
    print("\n📊 Configuring PAI tracker...")
    optim_args = {
        'params': model.parameters(),
        'lr': config['training']['learning_rate']
    }
    sched_args = {
        'mode': 'max',  # We're maximizing score
        'patience': 5
    }
    
    optimizer, scheduler = configure_transformer_tracker(
        model,
        torch.optim.AdamW,
        torch.optim.lr_scheduler.ReduceLROnPlateau,
        optim_args,
        sched_args
    )
    
    # Print PAI configuration
    print_pai_info()
    
    # 7. Training loop (PAI will control when to stop)
    print("\n🚀 Starting fine-tuning with PerforatedAI...")
    print("=" * 60)
    print("PAI will automatically determine when to stop training.")
    print("Look for PAI_LocalLlamaCoder/PAI_LocalLlamaCoder.png after training!")
    print("=" * 60 + "\n")
    
    epoch = 0
    training_complete = False
    
    while not training_complete and epoch < args.epochs:
        epoch += 1
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}")
        print(f"{'='*60}")
        
        # Training
        train_score, train_loss = train_epoch(model, train_loader, optimizer, device)
        print(f"Train Loss: {train_loss:.4f}, Train Score: {train_score:.4f}")
        
        # Add training score to PAI (optional but recommended)
        add_training_score(train_score)
        
        # Evaluation
        val_score, val_loss = evaluate(model, eval_loader, device)
        print(f"Val Loss: {val_loss:.4f}, Val Score: {val_score:.4f}")
        
        # CRITICAL: Add validation score and check for restructuring
        model, restructured, training_complete = add_validation_score_and_check_restructure(
            val_score,
            model
        )
        
        # Handle restructuring (when dendrites are added/incorporated)
        if restructured:
            print("\n📊 Model restructured! Reinitializing optimizer...")
            # Move to device again after restructuring
            if not config['model']['load_in_8bit']:
                model.to(device)
            
            # Reinitialize optimizer and scheduler
            optim_args = {
                'params': model.parameters(),
                'lr': config['training']['learning_rate']
            }
            optimizer, scheduler = configure_transformer_tracker(
                model,
                torch.optim.AdamW,
                torch.optim.lr_scheduler.ReduceLROnPlateau,
                optim_args,
                sched_args
            )
        
        # Check if training is complete
        if training_complete:
            print("\n✅ PAI determined training is complete!")
            print("Best model has been loaded automatically.")
            break
        
        # Demo mode - stop early
        if args.demo and epoch >= 3:
            print("\n⚠️  Demo mode: Stopping after 3 epochs")
            break
    
    # 8. Save the final model
    print("\n💾 Saving fine-tuned model...")
    output_dir = config['training']['output_dir']
    if args.demo:
        output_dir = f"{output_dir}/demo"
    
    final_path = f"{output_dir}/final"
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    
    print(f"\n✅ Fine-tuning complete! Model saved to: {final_path}")
    print(f"📊 PAI outputs saved to: PAI_LocalLlamaCoder/")
    print(f"📈 **IMPORTANT**: Check PAI_LocalLlamaCoder/PAI_LocalLlamaCoder.png for validation graph!")
    print("\nNext steps:")
    print(f"  python inference_code.py --model {final_path}")
    print(f"  python demo_interactive.py --model {final_path}")


if __name__ == "__main__":
    main()
