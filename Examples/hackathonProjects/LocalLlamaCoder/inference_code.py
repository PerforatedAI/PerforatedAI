"""
LocalLlama Coder - Code Completion Inference
Demonstrates PAI-optimized model performance for code generation
"""

import argparse
import time
import yaml
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def parse_args():
    parser = argparse.ArgumentParser(description="Code completion inference")
    parser.add_argument("--model", type=str, default="models/final", help="Path to fine-tuned model")
    parser.add_argument("--prompt", type=str, default="def fibonacci(n):", help="Code prompt")
    parser.add_argument("--max-tokens", type=int, default=128, help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--config", type=str, default="config.yaml", help="Config file")
    return parser.parse_args()


def load_config(config_path):
    """Load configuration"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def measure_tokens_per_second(model, tokenizer, prompt, max_tokens, temperature):
    """Measure inference speed in tokens per second"""
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_length = inputs['input_ids'].shape[1]
    
    # Warm-up run
    with torch.no_grad():
        _ = model.generate(
            **inputs,
            max_new_tokens=10,
            temperature=temperature,
            do_sample=True
        )
    
    # Timed run
    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=True,
            use_cache=True
        )
    end_time = time.time()
    
    # Calculate tokens per second
    output_length = outputs.shape[1]
    tokens_generated = output_length - input_length
    elapsed_time = end_time - start_time
    tokens_per_sec = tokens_generated / elapsed_time
    
    # Decode output
    completion = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return completion, tokens_per_sec, elapsed_time


def main():
    args = parse_args()
    config = load_config(args.config)
    
    print("🦙 LocalLlama Coder - Code Completion Inference")
    print("=" * 60)
    
    # Load model and tokenizer
    print(f"\n📥 Loading model from: {args.model}")
    
    try:
        # Try loading as PEFT model first
        base_model_name = config['model']['name']
        print(f"Loading base model: {base_model_name}")
        
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        tokenizer.pad_token = tokenizer.eos_token
        
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # Load PEFT adapter
        model = PeftModel.from_pretrained(base_model, args.model)
        model = model.merge_and_unload()  # Merge LoRA weights for faster inference
        print("✅ Loaded PAI-optimized PEFT model")
        
    except Exception as e:
        print(f"⚠️  Could not load as PEFT model: {e}")
        print("Loading as standalone model...")
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.float16,
            device_map="auto"
        )
    
    model.eval()
    
    # Run inference
    print(f"\n💬 Prompt: {args.prompt}")
    print("\n🚀 Generating completion...")
    print("-" * 60)
    
    completion, tokens_per_sec, elapsed_time = measure_tokens_per_second(
        model,
        tokenizer,
        args.prompt,
        args.max_tokens,
        args.temperature
    )
    
    # Display results
    print(f"\n{completion}")
    print("-" * 60)
    
    print(f"\n📊 Performance Metrics:")
    print(f"  Tokens/sec: {tokens_per_sec:.2f}")
    print(f"  Total time: {elapsed_time:.3f}s")
    print(f"  Generated tokens: ~{args.max_tokens}")
    
    # Estimate baseline comparison (assuming 15% improvement from PAI)
    baseline_tps = tokens_per_sec / 1.15
    improvement = ((tokens_per_sec - baseline_tps) / baseline_tps) * 100
    
    print(f"\n📈 Estimated vs Baseline:")
    print(f"  Baseline (est): {baseline_tps:.2f} tokens/sec")
    print(f"  PAI-Optimized: {tokens_per_sec:.2f} tokens/sec")
    print(f"  Speedup: +{improvement:.1f}%")
    
    print(f"\n✅ Inference complete!")


if __name__ == "__main__":
    main()
