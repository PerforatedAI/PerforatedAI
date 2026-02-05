# ImageNet
This file contains code to train a ResNet on the ImageNet dataset.

our huggingfacemodel was trained with the following command:

    CUDA_VISIBLE_DEVICES=0 python -m pdb train_perforated.py --model resnet18 --batch-size 32 --lr 0.0125 --val-resize-size 256 --val-crop-size 224 --train-crop-size 224 --full-dataset --data-path /home/rbrenner/Datasets/imagenet --convert-count 0 --dendrite-mode 1 --improvement-threshold 1 --candidate-weight-init-mult 0.1 --pai-forward-function relu
