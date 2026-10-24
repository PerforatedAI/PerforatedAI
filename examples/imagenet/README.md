# ImageNet

This folder contains code to train a on the ImageNet dataset starting from the official PyTorch training docs [here](https://github.com/pytorch/vision/tree/main/references/classification#mobilenetv3-large--small).

our huggingfacemodel was trained with the following command:

    CUDA_VISIBLE_DEVICES=0 python train_perforated.py --model resnet18 --batch-size 32 --lr 0.0125 --val-resize-size 256 --val-crop-size 224 --train-crop-size 224 --full-dataset --data-path /path/to/imagenet --convert-count 0 --dendrite-mode 1 --improvement-threshold 1 --candidate-weight-init-mult 0.1 --pai-forward-function relu

Results of retained drendrites for the model shown here:


![Clean Graph](clean_graph.png)


## Additional Files

 - resnet_prefc.py
   - This file enables a pre_fc layer to be added to a ResNet model so that when perforating during pretraining transfer learning can still be done since replacing the final FC layer would also replace the dendrites if only the FC layer is perforated.
 - efficientnet_prefc.py
   - Wrapper class that adds a perforable pre-FC layer before the classifier in EfficientNet models, enabling transfer learning while preserving dendrite connections.
 - train_fast_perforatedai_sweep.py
   - A lightweight training script to sweep hyperparameters without the full dataset before training on the full dataset.  
 - train_perforated_resnet.py
   - Generally the same training script but with defaults for resnet training filled in.
 - train_perforated_efficientnet.py
   - Training script for EfficientNet-V2 based on the official PyTorch EfficientNet-V2 recipe, using efficientnet_prefc.py.  The recommended command is in the docstring at the top of the file.
 - train_perforated_mobilenetv3.py
   - Training script for MobileNetV3 based on the official PyTorch MobileNetV3 recipe.  It also averages the weights of the top 3 checkpoints when switching to dendrite training.  The recommended command is in the docstring at the top of the file.
 - train_perforated_resnet_data_efficiency.py
   - An experiment that allows a user to decide what percent of the training data to actually train on.  This file is what was used to verify perforated models can surpass accuracy of baseline models with 25% less data.

## Known Limitations

 - `train_perforated.py` and `train_fast_perforatedai_sweep.py` don't support `--model-ema` yet.  With it enabled they also pass the EMA model to `add_validation_score`, which stops the run.  The flag is off by default and isn't needed for the ResNet commands above.
