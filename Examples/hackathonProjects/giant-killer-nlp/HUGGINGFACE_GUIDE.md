# Pushing to Hugging Face - Quick Guide

## Prerequisites

1. **Hugging Face Account**: Sign up at https://huggingface.co/join
2. **Install huggingface_hub**:
   ```bash
   pip install huggingface_hub
   ```
3. **Login to Hugging Face**:
   ```bash
   huggingface-cli login
   ```
   Enter your access token when prompted (create one at https://huggingface.co/settings/tokens)

## Step-by-Step Instructions

### Option 1: Direct Push (Recommended)

```bash
# Push your best model to Hugging Face
python push_to_hf.py --repo_id YOUR_USERNAME/dendritic-bert-tiny-toxicity --checkpoint checkpoints/best_model.pt
```

Replace `YOUR_USERNAME` with your actual Hugging Face username.

### Option 2: Prepare Files Locally First

```bash
# Prepare files locally without pushing
python push_to_hf.py --repo_id YOUR_USERNAME/dendritic-bert-tiny-toxicity --checkpoint checkpoints/best_model.pt --local_only

# Review the files in ./hf_model/
# Then push manually:
cd hf_model
git init
git add .
git commit -m "Initial commit"
git remote add origin https://huggingface.co/YOUR_USERNAME/dendritic-bert-tiny-toxicity
git push -u origin main
```

### Option 3: Using a Token Directly

```bash
# If you prefer not to use huggingface-cli login
python push_to_hf.py --repo_id YOUR_USERNAME/dendritic-bert-tiny-toxicity --checkpoint checkpoints/best_model.pt --token YOUR_HF_TOKEN
```

## What Gets Uploaded

The script will upload:
- ✅ Model weights (PyTorch)
- ✅ Model configuration
- ✅ Tokenizer files
- ✅ Model card (README.md) with metrics and usage examples
- ✅ Training configuration (config.yaml)
- ✅ Requirements file
- ✅ Training summary
- ✅ Threshold optimization report

## After Uploading

1. Visit your model page: `https://huggingface.co/YOUR_USERNAME/dendritic-bert-tiny-toxicity`
2. Verify all files uploaded correctly
3. Update the README if needed (edit MODEL_CARD.md and re-push)
4. Test your model:
   ```python
   from transformers import pipeline
   classifier = pipeline('text-classification', model='YOUR_USERNAME/dendritic-bert-tiny-toxicity')
   result = classifier("This is a test comment")
   print(result)
   ```

## Troubleshooting

### Error: "Repository not found"
- Make sure you're logged in: `huggingface-cli login`
- Check your username is correct

### Error: "Authentication failed"
- Your token might not have write access
- Create a new token with write permissions at https://huggingface.co/settings/tokens

### Error: "Checkpoint not found"
- Verify your checkpoint path:
  ```bash
  ls checkpoints/
  ```
- Available options: `best_model.pt` or `final_model.pt`

### Want to push a different checkpoint?
```bash
python push_to_hf.py --repo_id YOUR_USERNAME/dendritic-bert-tiny-toxicity --checkpoint checkpoints/final_model.pt
```

## Model Naming Convention

Suggested names for your repository:
- `dendritic-bert-tiny-toxicity` (recommended)
- `bert-tiny-toxic-detection`
- `efficient-toxicity-classifier`
- `perforated-bert-toxicity`

Choose a descriptive name that indicates:
1. The base model (BERT-Tiny)
2. The task (toxicity detection)
3. The technique (dendritic/perforated) - optional

## Example: Complete Workflow

```bash
# 1. Install dependencies
pip install huggingface_hub

# 2. Login to Hugging Face
huggingface-cli login

# 3. Push your model (replace with your username)
python push_to_hf.py --repo_id johndoe/dendritic-bert-tiny-toxicity --checkpoint checkpoints/best_model.pt

# 4. Success! Visit https://huggingface.co/johndoe/dendritic-bert-tiny-toxicity
```

## Updating Your Model

To update after pushing:

```bash
# Make changes to MODEL_CARD.md or retrain model
# Then push again:
python push_to_hf.py --repo_id YOUR_USERNAME/dendritic-bert-tiny-toxicity --checkpoint checkpoints/best_model.pt
```

The script will update existing files.

---

**Need Help?**
- Hugging Face Documentation: https://huggingface.co/docs/hub
- Hub CLI Reference: https://huggingface.co/docs/huggingface_hub
