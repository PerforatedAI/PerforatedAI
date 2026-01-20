"""
LocalLlama Coder - Interactive Coding Demo
Real-time code completion interface demonstrating PAI optimization
"""

import argparse
import yaml
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from peft import PeftModel


def parse_args():
    parser = argparse.ArgumentParser(description="Interactive coding demo")
    parser.add_argument("--model", type=str, default="models/final", help="Path to fine-tuned model")
    parser.add_argument("--config", type=str, default="config.yaml", help="Config file")
    return parser.parse_args()


def load_config(config_path):
    """Load configuration"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    args = parse_args()
    config = load_config(args.config)
    
    print("🦙 LocalLlama Coder - Interactive Demo")
    print("=" * 60)
    print("Type your code and press Enter. Type 'quit' to exit.")
    print("=" * 60)
    
    # Load model
    print("\n📥 Loading PAI-optimized model...")
    
    try:
        base_model_name = config['model']['name']
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        tokenizer.pad_token = tokenizer.eos_token
        
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        model = PeftModel.from_pretrained(base_model, args.model)
        model = model.merge_and_unload()
        
    except Exception as e:
        print(f"Loading standalone model...")
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.float16,
            device_map="auto"
        )
    
    model.eval()
    print("✅ Model loaded!\n")
    
    # Interactive loop
    while True:
        try:
            # Get user input
            prompt = input("\n💬 Code: ")
            
            if prompt.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break
            
            if not prompt.strip():
                continue
            
            # Prepare inputs
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            # Create streamer for real-time output
            streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
            
            print("\n🚀 Completion:")
            print("-" * 60)
            
            # Generate with streaming
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=config['inference']['max_new_tokens'],
                    temperature=config['inference']['temperature'],
                    top_p=config['inference']['top_p'],
                    do_sample=config['inference']['do_sample'],
                    streamer=streamer,
                    use_cache=config['inference']['use_cache']
                )
            
            print("-" * 60)
            
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n⚠️  Error: {e}")
            continue


if __name__ == "__main__":
    main()
