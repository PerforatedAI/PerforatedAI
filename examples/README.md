# Dendritic Augmentation Examples

This folder contains working examples of how to add dendrites to existing PyTorch systems. Each example contains full code and a README for running it with both the original code and the PAI code. Each README reports the results of running the original repository alongside the results we got with Dendritic Augmentation. Original repositories are linked, along with the date we checked them out. To find modified files, search for files with `_perforatedai.py` in their names.

General instructions for adding PAI to a system can be found in the [API](https://github.com/PerforatedAI/PerforatedAI/tree/main/api) directory. The specific examples of where to put each function and how to use the customization functions can be found here to best implement the system into your own training program.

> **New to Perforated?** Start with the [MNIST example](baseExamples/mnist) - it is the smallest end-to-end integration and doubles as the template for contributing new examples.

## Index

### Base Examples

Foundational, first-party integrations across common architectures.

| Example                                                                                    | What it shows                                                                                         |
| ------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------- |
| [MNIST](baseExamples/mnist)                                                                | The canonical integration and the template to copy when contributing a new example. Start here.       |
| [ResNet](baseExamples/resnet)                                                              | ResNet-18 on Food-101 combined with knowledge distillation.                                           |
| [Transformer](baseExamples/transformer)                                                    | Head-to-head comparison of a dendritic vs. vanilla Transformer language model.                        |
| [YOLO (Pascal)](baseExamples/yolo-pascal)                                                  | YOLOv11n object detection made more accurate with dendrites.                                          |
| [PyTorch-UNet](baseExamples/Pytorch-UNet)                                                  | U-Net image segmentation (fork of [milesial/Pytorch-UNet](https://github.com/milesial/Pytorch-UNet)). |
| [Segmentation Resolution](baseExamples/segmentation-resolution/Human-Segmentation-PyTorch) | Human-segmentation experiments across input resolutions.                                              |
| [scRNA Cell Type](baseExamples/scrna_celltype)                                             | Single-cell RNA-seq cell-type annotation with dendritic transformers.                                 |

### Library & Framework Integrations

Using PAI inside popular training frameworks.

| Example                                                | What it shows                                                                                           |
| ------------------------------------------------------ | ------------------------------------------------------------------------------------------------------- |
| [Hugging Face](libraryExamples/huggingface)            | Transformers with dendritic layers - MNIST, BERT classification, and a ViT-Tiny classifier on RVL-CDIP. |
| [PyTorch Lightning](libraryExamples/pytorch_lightning) | Integrating PAI into a PyTorch Lightning training loop (MNIST).                                         |

### Reinforcement Learning

| Example                                            | What it shows                                         |
| -------------------------------------------------- | ----------------------------------------------------- |
| [TD3 (Cheetah)](reinforcementLearning/td3_cheetah) | TD3 continuous control on the DeepMind Control Suite. |
| [DQN (MinAtar)](reinforcementLearning/dqn_minatar) | DQN on MinAtar's miniaturized Atari 2600 games.       |

### Challenge Examples

Larger or harder reproductions.

| Example                              | What it shows                                   |
| ------------------------------------ | ----------------------------------------------- |
| [nanoGPT](challengeExamples/nanoGPT) | Reproducing Perforated AI's results on nanoGPT. |

### ImageNet

| Example                                    | What it shows                                               |
| ------------------------------------------ | ----------------------------------------------------------- |
| [ImageNet](imagenet)                       | Training a ResNet on the ImageNet dataset.                  |
| [ImageNet Pretrained](imagenet_pretrained) | Using our ImageNet-pretrained models (currently ResNet-18). |

### Additional Examples

Experimental and domain-specific integrations.

| Example                                                                         | What it shows                                                                       |
| ------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------- |
| [KAN](additionalExamples/KAN)                                                   | Converting and experimenting with Kolmogorov–Arnold Networks via `perforate_model`. |
| [Fire Detection](additionalExamples/fire_detection_pai_experiments)             | Wildfire prediction from satellite-derived vegetation and temperature data.         |
| [Hallucination Classification](additionalExamples/hallucination_classification) | Detecting hallucinated answers in medical QA pairs (MedHallu).                      |

### Hackathon Projects

Community submissions from hackathons - a large and growing collection covering many additional architectures and domains. These have not necessarily been verified by us, so use them as examples at your own risk. See the [Hackathon Projects index](hackathonProjects) for the full list.

## Contributing

We welcome any contributions to this folder. Please look at the [MNIST example](baseExamples/mnist) for how to format a contribution.

## Licensing

Some example folders are forks of other repos. Where those repos contain license files, the internal licenses only apply to the pre-existing code from the original folders - not to any upstream code, or to any `_pai`/`_perforatedai` files within that folder.
