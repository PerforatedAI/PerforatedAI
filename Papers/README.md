# Papers and Summaries

This page is to accumulate papers on adding dendrites to neural network projects. We will only be accumulating methods adding dendrites to traditional deep learning applications, exceptions such as spiking networks and hardware projects will not be included.  Submit a PR if you would like to add to these resources.

## Overview

| Paper | Summary | Differentiator | Gradient Descent? | Dendrites on Multiple Neuron Layers |
|-------|---------|----------------|-----|-----------------------|
| [The Cascade Correlation Learning Architecture](https://proceedings.neurips.cc/paper/1989/file/69adc1e107f7f7d035d7baf04342e1ca-Paper.pdf) | Introduces the correlation learning rule for dendrites | Correlation learning | No | No |
| [Morphological Perceptrons with Dendritic Structure](https://ieeexplore.ieee.org/document/1206618) | Dendrites are more parameter efficient | Dendrites create hypercubes | No | No |
| [Efficient Training for Dendrite Morphological Neural Networks](https://www.sciencedirect.com/science/article/pii/S0925231213010916) | With enough dendrites one neuron can perfectly classify any training dataset | Dendrites create hypercubes | No | No |
| [Dendritic Neuron Model With Effective Learning Algorithms for Classification, Approximation, and Prediction](https://ieeexplore.ieee.org/document/8409490) | Dendrites are effective for binary classification | Multiple learning approaches | No | No |
| [Drawing Inspiration from Biological Dendrites to Empower Artificial Neural Networks](https://www.sciencedirect.com/science/article/abs/pii/S0959438821000544) | Review Paper | N/A | N/A | N/A |
| [Learning on Tree Architectures Outperforms a Convolutional Feedforward Network](https://www.nature.com/articles/s41598-023-27986-6) | Dendrites are more parameter efficient | First paper with dendrites for CNNs | Yes | No |
| [Dendrites Endow Artificial Neural Networks with Accurate, Robust and Parameter-Efficient Learning](https://www.nature.com/articles/s41467-025-56297-9) | Dendrites are more parameter efficient | Dendritic receptive fields | Yes | No |
| [Perforated Backpropagation: A Neuroscience Inspired Extension to Artificial Neural Networks](https://arxiv.org/pdf/2501.18018) | PB dendrites increase accuracy | Correlation learning and Perforated Backpropagation | No | Yes |
| [Exploring the Performance of Perforated Backpropagation through Further Experiments](https://arxiv.org/pdf/2501.18018) | PB dendrites achieve compression | Correlation learning and Perforated Backpropagation | No | Yes |



## The Cascade Correlation Learning Architecture - 1989
[The Cascade Correlation Learning Architecture](https://proceedings.neurips.cc/paper/1989/file/69adc1e107f7f7d035d7baf04342e1ca-Paper.pdf) is the first to introduce the correlation learning rule to differentiate dendrite nodes from neuron nodes.  This paper is from 1989 and was introduced as an alternative to Gradient Descent rather than a mechanism to integrate with it.  It also does not discuss dendrites itself, but is the inspiration for the cascading architecture used in this repository.

## Morphological Perceptrons with Dendritic Structure - 2003

[Morphological Perceptrons with Dendritic Structure](https://ieeexplore.ieee.org/document/1206618) uses dendritic branches to define geometric regions in the input space, often represented as hypercubes or other shapes, through excitatory and inhibitory connections, allowing the neuron to recognize complex spatial patterns. Unlike traditional neural networks that rely on global weight adjustments and backpropagation, this model employs localized, competitive learning rules to enclose target input patterns within these dendritic activation regions. This biologically inspired approach mimics real dendritic integration, providing a more hardware-efficient and interpretable way to achieve nonlinear classification without needing multiple layers.

## Efficient Training for Dendrite Morphological Neural Networks - 2014

[Efficient Training for Dendrite Morphological Neural Networks](https://www.sciencedirect.com/science/article/pii/S0925231213010916) introduces the addition of dendrites to morphological perceptrons which can train dendrites to perfectly classify any training dataset. The algorithm instantiates a neuron with dendrites for a set of input data by iteratively creating hypercubes which encapsulate possible input patterns. First a hypercube is generated which encapsulates all the data. Then if the training data inside a hypercube entirely belongs to a single class the hypercube is saved. If the training data is not entirely within a single class, the hypercube is iteratively split into smaller hypercubes. Once this process is complete the neuron is given a set of dendritic branches corresponding to each hypercube where the branches have frozen weights such that a particular branch is only active if a datapoint is within its associated hypercube. The authors showed this algorithm could perform well on basic pattern recognition and even be applied to image processing tasks outperforming simple multi-layer perceptrons.

## Dendritic Neuron Model With Effective Learning Algorithms for Classification, Approximation, and Prediction - 2018
[Dendritic Neuron Model With Effective Learning Algorithms for Classification, Approximation, and Prediction](https://ieeexplore.ieee.org/document/8409490) uses traditional neurons but strays from traditional neural network training by not using backpropagation as its training paradigm. Instead, the authors ran experiments using six learning algorithms: biogeography-based optimization, particle swarm optimization, genetic algorithm, ant colony optimization, evolutionary strategy, and population-based incremental learning. 

## Drawing Inspiration from Biological Dendrites to Empower Artificial Neural Networks - 2021

[Drawing Inspiration from Biological Dendrites to Empower Artificial Neural Networks](https://www.sciencedirect.com/science/article/abs/pii/S0959438821000544) reviews additional papers.

## Learning on Tree Architectures Outperforms a Convolutional Feedforward Network - 2023
[Learning on Tree Architectures Outperforms a Convolutional Feedforward Network](https://www.nature.com/articles/s41598-023-27986-6) is the first to add dendrites to convolutional neurons. The dendrites are trained with Gradient Descent, and the results show that a dendritic model can be created for CIFAR-10 that is much more parameter-efficient than other convolutional models, such as LeNet, while achieving similar accuracy.

## Dendrites Endow Artificial Neural Networks with Accurate, Robust and Parameter-Efficient Learning - 2025

[Dendrites Endow Artificial Neural Networks with Accurate, Robust and Parameter-Efficient Learning](https://www.nature.com/articles/s41467-025-56297-9) This paper builds an MLP architecture with one hidden layer of neurons with dendrites before an output layer without dendrites.  The dendritic architecture explores more biologically realistic receptive field formats of where dendrites can get input from the input image.  The experiments show dendritic architectures are significantly more parameter efficient than a traditional MLP based approach.

## Perforated Backpropagation: A Neuroscience Inspired Extension to Artificial Neural Networks - 2025

[Perforated Backpropagation: A Neuroscience Inspired Extension to Artificial Neural Networks](https://arxiv.org/pdf/2501.18018) is the original paper by Perforated AI.  This is the first to instantiate dendrites with a mechanism compatible with modern deep learning frameworks, while differentiating from the standard Gradient Descent algorithm for training.  Dendrites in Perforated Backpropagation learn with a correlation learning rule to identify areas of the input space where patterns are causing problems for the feature classification of individual neurons.  Once added they exist inside the system during the forward pass, but outside of the system during the backwards pass by being ignored during the Gradient Descent process.  In this way if a network has N neurons calculating N features, the architecture still only calculates N features as defined by error communication, but each feature is also enabled to be more complex by the support of the added dendrites.  This paper shows that by adding dendrites to state-of-the-art methods on a handful of applications, the accuracies can be improved.

## Exploring the Performance of Perforated Backpropagation through Further Experiments - 2025

[Exploring the Performance of Perforated Backpropagation through Further Experiments](https://arxiv.org/pdf/2501.18018) is the second paper by Perforated AI. This paper focuses on creating smaller, more efficient models by reducing original network sizes then adding dendrites and training on the same datasets. It examines more common benchmarks and architectures, and is the outcome of collaborative experiments run during a hackathon. Additionally, it highlights industry impact by deploying one of the trained models on Google Cloud and reporting the changes in cost and speed between running the original model and the dendritic model.
