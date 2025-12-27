# Dendritic NN Block for Edge Impulse

This repository implements a neural network block for Edge Impulse which leverages dendritic optimization. To see details on how to compile and push this block to Edge Impulse, follow the instructions from the [original repository](https://github.com/edgeimpulse/example-custom-ml-block-pytorch/tree/master). This block was created for the 2025 Edge Impulse Hackathon. A submission video describing the project is available [here](https://drive.google.com/file/d/1vUdMBLijVm4c0Sq_RrBlrPgo-rGQZHWZ/view?usp=sharing).

## What is Dendritic Optimization?

The original artificial neuron was proposed in 1943, drawing on neuroscience research dating back to the 1860s. Since then, backpropagation was introduced, and there have been significant advances in hardware, optimizers, data curation, and architectures, while the core building block has remained fundamentally the same. Interestingly, for 70 of the last 80 years, neuroscience continued to support this original design. However, modern neuroscience now understands that the perceptron misses a critical piece of biological intelligence: the decision-making performed by a neuron's dendrites. Dendritic optimization leverages these ideas to augment artificial neurons with dendrite nodes, enabling ML practitioners to achieve smarter, smaller, and cheaper models on the same datasets. Experiments frequently show 10-20% reduced error rates after dendritic optimization as well as the ability to compress models by up to 90% without loss in accuracy.  By enabling users to develop smaller models with equal accuracy this also enables AI to be built with 90% reduced carbon footprint without impact on end users.  For further details about this research, a selection of papers can be found [here](https://github.com/PerforatedAI/PerforatedAI/tree/main/Papers).

## This Project

This project first explored the improvements dendritic optimization could achieve on the model in the [keyword spotting tutorial](https://docs.edgeimpulse.com/tutorials/end-to-end/keyword-spotting), and then created a public Edge Impulse block to enable anyone to leverage this capability on their own Edge Impulse projects. 

## Our Experiments

For details on our experiments, please view the [W&B report](https://wandb.ai/perforated-ai/Dendritic%20Edge%20Impulse%20Audio%20-%20Combo/reports/Edge-Impulse-Keyword-Spotting--VmlldzoxNTIxNjE5Ng?accessToken=3lm4jm5f9npsu45vs180ybo6150ed4gnhos9rrkk6seqb4bmf458me28seynu0xb) of the 800 trials we ran while sweeping hyperparameters for this application.

## This Repository

This repository replaces the PyTorch script from the example PyTorch block with our custom script. It updates the hyperparameter settings to enable users to experiment with all of the hyperparameters we swept over. It also compiles the final dendritic models in ONNX format to be used in exactly the same way as the original block. This is a plug-and-play Impulse Block allowing users to use dendritic optimization on any Edge Impulse project that uses audio data. As an open-source project, it enables users to make required adjustments to work with additional data formats. Additionally, working with the Edge Impulse team, this block provides a starting point to extend the default Edge Impulse NN Classifier block with dendritic optimization, empowering all Edge Impulse users to achieve improved outcomes on any project by checking a single checkbox.

## Example Output Graphs

The image below is one selected output graph from a training run. Although the high oscillations cause difficulty in seeing improvements from dendrites 1 and 2, dendrites 3 and 4 show clear improvements above the previous architectures.

<img src="./Dendritic NN Impulse Block Training.png" alt="Example training output graph." width="300" height="200">

Questions for edge impulse:
- How are the accuracy scores?
- How are the augmentation methods separated out?