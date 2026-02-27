# ImageNet Pretrained

This folder shows how to use our models that have been pretrained on ImageNet. Right now it is just a resnet-18. An example script can be run with:

    python train_flowers_from_hf.py --hf-repo-id "perforated-ai/resnet-18-perforated"

## train_from_hf_sweep.py (multi-run sweep)

Runs multiple training sessions from a HuggingFace repo ID and reports CSV stats + latency. Defaults to 7 runs.

    python train_from_hf_sweep.py \
      --hf-repo-id "perforated-ai/resnet-18-perforated" \
      --dataset flowers102 \
      --num-runs 7

You can also use non-perforated HF repos (e.g., a transformers or torchvision fallback model):

    python train_from_hf_sweep.py \
      --hf-repo-id "microsoft/resnet-18" \
      --dataset flowers102 \
      --num-runs 7

## train_perforatedai_hf.py (single-run + HF upload/download)

Single training session using the PerforatedAI pipeline. Supports loading from HF or uploading a locally loaded model to HF.

Load from HuggingFace and run a single session:

    python train_perforatedai_hf.py \
      --dataset flowers102 \
      --model resnet18 \
      --perforatedai \
      --hf-mode 2 \
      --hf-repo-id "perforated-ai/resnet-18-perforated"

Upload a locally loaded model to HuggingFace (uses the current model definition and weights):

    python train_perforatedai_hf.py \
      --dataset flowers102 \
      --model resnet18 \
      --perforatedai \
      --hf-mode 1 \
    --hf-repo-id "your-username/your-model-name" \
    --pretrained-path ./pretrained/best_model.pt

If you want to upload a different local checkpoint, pass a custom path (must end in .pt):

    python train_perforatedai_hf.py \
    --dataset flowers102 \
    --model resnet18 \
    --perforatedai \
    --hf-mode 1 \
    --hf-repo-id "your-username/your-model-name" \
    --pretrained-path /path/to/your_checkpoint.pt

## Using the pretrained ResNet-18 in your own project

First install the repo to access the base model that includes the pre-fc layer:

    pip install perforatedai

Then add the following lines:

    from perforatedai import utils_perforatedai as UPA
    from perforatedai import library_perforatedai as LPA

    # Create base model architecture
    base_model = torchvision.models.get_model('resnet18', weights=None, num_classes=1000)
    # Convert the standard resnet to our new architecture
    model = LPA.ResNetPAIPreFC(base_model)
    # Load weights from HuggingFace
    model = UPA.from_hf_pretrained(model, args.hf_repo_id)