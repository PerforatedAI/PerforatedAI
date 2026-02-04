# ImageNet Pretrained

This folder shows how to use our models that have been pretrained on ImageNet. Right now it is just a resnet-18.  An example scipt can be run with:

    python train_flowers_from_hf.py --hf-repo-id "perforated-ai/resnet-18-perforated"

To use our resnet-18 in your own project first install our repo to access our base model which has the pre-fc layer added:

    pip install perforatedai

Then add the following lines:

    from perforatedai import utils_perforatedai as UPA
    from perforatedai import library_perforatedai as LPA
        
    # Create base model architecture
    base_model = torchvision.models.get_model('resnet18', weights=None, num_classes=1000)
    # Convert the standard resnet to our new arcitechture
    model = LPA.ResNetPAIPreFC(base_model)
    # Load weights from HuggingFace
    model = UPA.from_hf_pretrained(model, args.hf_repo_id)