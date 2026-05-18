# Perforated NN Block for Edge Impulse

This repository implements a neural network block for Edge Impulse which leverages dendritic optimization. For easy install click the button below to open in codespaces then follow the listed instructions.  If this is your first time using Edge Impulse and you do not already have a classificaiton project you want to optimize, use the more detailed walkthrough [here](FULL_TUTORIAL.md).

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/PerforatedAI/PerforatedAI?devcontainer_path=Examples%2FhackathonProjects%2Fexample-custom-ml-block-pytorch%2F.devcontainer%2Fdevcontainer.json)

 - Use default settings and click "Create codespace"
 - At the bottom center of the window that loads is the terminal to run commands
 - In the teriminal run the following to install the edge-ipulse-cli, login to your account, and upload the block.

```
    git checkout nn_customize 
    npm install -g edge-impulse-cli --silent
    cd Examples/hackathonProjects/perforated-impulse-nn-block/
    edge-impulse-blocks init
    edge-impulse-blocks push
```

Now that your block is uploaded you can do the following:
 - Open your project with a classifier block
 - Click "Create Impulse"
 - Under your classifier block click "add learning block"
 - Select the name of the block you just uploaded
 - Click "Save Impulse"
 - Open your classificaiton block and the new perforated block in two tabs
 - Recreate your exact settings in the new perforated block
   - This usually means to check learning rate and back size as well as network definition
   - For network definition select each layer type and type with text the settings for the layer
     - These are comma-separated values and the text will show up in order to replace with numbers
   - If a layer that is in your classifier block does not exist in the perforated block it is done automatically
   - The final fully connected layer also is done automatically, no need to edit the text in the "settings" window for that block.
 - Click "Save & train" to run the new block!
 - Check your results
   - In the window it will show the scores before and after dendrites were added
   - Check if performance improved
   - Check if this performance actually beat your original nn performance
 - Try other hyperparameter settings
   - Often a smaller model with dendrites will outperform a larger nn classifier
 - Please let us know how it goes for you!

## What is Dendritic Optimization?

The original artificial neuron was proposed in 1943, drawing on neuroscience research dating back to the 1860s. Since then, backpropagation was introduced, and there have been significant advances in hardware, optimizers, data curation, and architectures, while the core building block has remained fundamentally the same. Interestingly, for 70 of the last 80 years, neuroscience continued to support this original design. However, modern neuroscience now understands that the perceptron misses a critical piece of biological intelligence: the decision-making performed by a neuron's dendrites. Dendritic optimization leverages these ideas to augment artificial neurons with dendrite nodes, enabling ML practitioners to achieve smarter, smaller, and cheaper models on the same datasets. Experiments frequently show 10-20% reduced error rates after dendritic optimization as well as the ability to compress models by up to 90% without loss in accuracy.  By enabling users to develop smaller models with equal accuracy this also enables AI to be built with 90% reduced carbon footprint without impact on end users.  For further details about this research, a selection of papers can be found [here](https://github.com/PerforatedAI/PerforatedAI/tree/main/Papers).

## This Project

This project first explored the improvements dendritic optimization could achieve on the model in the [keyword spotting tutorial](https://docs.edgeimpulse.com/tutorials/end-to-end/keyword-spotting){:target="_blank"}, and then created a public Edge Impulse block to enable anyone to leverage this capability on their own Edge Impulse projects. 

## Our Experiments

For details on our experiments, please view the [W&B report](https://wandb.ai/perforated-ai/Dendritic%20Edge%20Impulse%20Audio%20-%20Combo/reports/Edge-Impulse-Keyword-Spotting--VmlldzoxNTIxNjE5Ng?accessToken=3lm4jm5f9npsu45vs180ybo6150ed4gnhos9rrkk6seqb4bmf458me28seynu0xb){:target="_blank"} of the 800 trials we ran while sweeping hyperparameters for this application.

## This Repository

This repository replaces the PyTorch script from the example PyTorch block with our custom script. It updates the hyperparameter settings to enable users to experiment with all of the hyperparameters we swept over. It also compiles the final dendritic models in ONNX format to be used in exactly the same way as the original block. This is a plug-and-play Impulse Block allowing users to use dendritic optimization on any Edge Impulse project that uses audio data. As an open-source project, it enables users to make required adjustments to work with additional data formats. Additionally, working with the Edge Impulse team, this block provides a starting point to extend the default Edge Impulse NN Classifier block with dendritic optimization, empowering all Edge Impulse users to achieve improved outcomes on any project by checking a single checkbox.

# Sweep Results

Over 800 sweeps dendritic models consistently showed improved performance at any given parameter count.  This graph shows the best models for dendritic and traditional architectures as parameter count goes up across the 800 experiments.

<img src="./Edge Impulse Sweep.png" alt="Edge Impulse Sweep." width="300" height="200">

## Example Output Graphs

The image below is one selected output graph from a training run. Although the high oscillations cause difficulty in seeing improvements from dendrites 1 and 2, dendrites 3 and 4 show clear improvements above the previous architectures.

<img src="./Dendritic NN Impulse Block Training.png" alt="Example training output graph." width="300" height="200">

Questions for edge impulse:
- How are the accuracy scores?
- How are the augmentation methods separated out?
