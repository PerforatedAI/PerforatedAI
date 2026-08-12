# Transfer Learning

This folder shows how to use perforation during transfer learning including out models that have been pretrained on ImageNet.

    python train_flowers_from_hf.py --hf-repo-id "perforated-ai/resnet-18-perforated-cascor"

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
      --model resnet18 \
      --perforatedai \
      --hf-mode 1 \
    --hf-repo-id "your-username/your-model-name" \
    --pretrained-path ./pretrained/best_model.pt

## train_from_hf_wandb_sweep_resnet.py (W&B hyperparameter sweep — ResNet)

Runs a W&B hyperparameter search over ResNet model variants and training configurations. Unlike `train_from_hf_sweep.py` which repeats the same config, this explores different hyperparameter combinations to find the best settings for a given dataset.

Supported models: `resnet-18-perforated-cascor-pretrained`, `resnet-18-perforated-cascor-fc`, `resnet-18-perforated-cascor-pre-fc`, `resnet-18-perforated-cascor-hf-fc`, `resnet-34`

Single training run:

    python train_from_hf_wandb_sweep_resnet.py \
      --model resnet-18-perforated-cascor-pretrained \
      --dataset flowers102

Initialize and run a full W&B sweep:

    python train_from_hf_wandb_sweep_resnet.py \
      --sweep-dataset flowers102 \
      --sweep-count 300

## train_from_hf_wandb_sweep_effecientnet.py (W&B hyperparameter sweep — EfficientNet)

Same W&B sweep structure as the ResNet version but for EfficientNet models. Uses torchvision ImageNet pretrained weights directly — no HuggingFace model loading. PAI is applied only to the classifier head in the perforated variant.

Supported models: `efficientnet-b4-perforated-fc`, `efficientnet-b5`

Single training run:

    python train_from_hf_wandb_sweep_effecientnet.py \
      --model efficientnet-b4-perforated-fc \
      --dataset flowers102

Initialize and run a full W&B sweep:

    python train_from_hf_wandb_sweep_effecientnet.py \
      --sweep-dataset flowers102 \
      --sweep-count 300

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
