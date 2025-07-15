# Perforated AI
Perforated AI is the Dendritic Intelligence Company, offering a novel algorithm to create artificial dendrites for PyTorch neural networks that are distinct from artificial neurons.  However, there is additional research in the field building neural models with a dendritic architectures.  These models are similar to ours in terms of architecture design, but the new nodes are still trained with gradient desecent.  Even so, these models are acheiving outstanding results and many do not have open source implimentations.  We release this open source code to support research into dendritic models.

Other dendritic AI repositories we have found simply enable you to run their code on their models.  The main differentiator for this open source implimentation is that with under an hour of coding you can add dendrites to any existing PyTorch Model.

Additional details can be found in our [API](https://github.com/PerforatedAI/PerforatedAI-API) and [examples repository](https://github.com/PerforatedAI/PerforatedAI-Examples).

## Comparison

We have only run preliminary experiments with this open source implimentation, but the following are our results.

### DSN BERT

DSN BERT is reproducing the results of our [hackathon winnners](https://www.perforatedai.com/natural-language-processing-3-25) experimetns with DSN BERT on the IMDB Dataset.
![BERT](BERT.png "BERT")

### ResNet 

ResNet is a new example running a ResNet 18 model on the EMNIST Dataset

![ResNet](ResNet.png "ResNet")

### Basic CNN

The CNN example is running the default PyTorch mnist example, included here, on the MNIST dataset.

![CNN](CNN.png "ResNet")
