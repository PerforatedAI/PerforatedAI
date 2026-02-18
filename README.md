<p align="center">
  <img src="./assets/logo.png" width="600" alt="Perforated AI" />
</p>

<p align="center">
<a href="https://pypi.python.org/pypi/perforatedai"><img src="https://img.shields.io/pypi/v/perforatedai" /></a>
<a href="https://pypi.python.org/pypi/perforatedai"><img src="https://img.shields.io/pypi/pyversions/perforatedai" /></a>
<a href="https://github.com/PerforatedAI/PerforatedAI"><img src="https://img.shields.io/github/stars/PerforatedAI/PerforatedAI" /></a>
</p>

Add biologically-inspired dendritic optimization to your PyTorch neural networks with just a few lines of code. Perforated AI (PAI) automatically enhances model accuracy in a highly parameter-efficient manner by perforating the architecture with dendrites.

<br>

**Key Innovation:** Unlike traditional neural networks that use simple point neurons, PAI adds dendritic branches that enable neurons to learn complex input patterns. Depending on your use case, PAI can deliver:
- **Up to 40% accuracy improvements** on challenging tasks
- **Up to 90% compression** without sacrificing accuracy
- **50% reduction in data requirements** for similar accuracy scores

All with **automatic network conversion** and minimal code changes.

&nbsp;

# Documentation

See the [API Documentation](./API/README.md) for detailed integration instructions, the [Customization Guide](./API/customization.md) for advanced configuration options, and [Output Guide](./API/output.md) for understanding training visualizations.

For research on dendritic methods in deep learning, see our [Papers](./Papers) collection which includes comparisons of different approaches and our published work.

&nbsp;

# Quickstart

Add dendritic learning to your PyTorch models in minutes.

## Install the perforatedai library

```bash
pip install -e .
```

## Add dendrites to your neural network

In your training script, import PAI utilities and wrap your model initialization:

```python
from perforatedai import globals_perforatedai as GPA
from perforatedai import utils_perforatedai as UPA
import torch

# Initialize your model
model = YourModel()

# Automatically add dendritic capabilities
model = UPA.initialize_pai(model)

# Setup optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
GPA.pai_tracker.set_optimizer_instance(optimizer)

# Train your model until dendrite optimization completes
while True:
    train(model, optimizer)
    val_score = validate(model)
    
    # Report validation score and handle model restructuring
    model, restructured, training_complete = GPA.pai_tracker.add_validation_score(val_score, model)
    model = model.to(device)  # Re-send to device after potential restructuring
    
    if training_complete:
        # Training is complete - best model has been loaded
        break
    elif restructured:
        # Model was restructured with new dendrites - reinitialize optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        GPA.pai_tracker.set_optimizer_instance(optimizer)
```

PAI automatically:
- Perforates model with dendrites for selected modules
- Identifies when to add dendrites based on validation scores
- Adjusts learning rates for optimal dendritic optimization
- Generates visualization graphs showing accuracy improvements

&nbsp;

# Examples

PAI works with many popular architectures and frameworks. Check out our [Examples](./Examples) folder for complete implementations:

## Base Examples
- **[MNIST](./Examples/baseExamples/mnist)** - Classic computer vision with dendritic enhancement
- **[TD3 Reinforcement Learning](./Examples/reinforcementLearning)** - TD3 with dendrites for continuous control

## Advanced Examples
- **[ImageNet ResNet-18](./Examples/imagenet)** - ResNet-18 with dendritic optimization ([Pretrained Model on Hugging Face](https://huggingface.co/perforated-ai/resnet-18-perforated))
- **[Edge Impulse Block](./Examples/hackathonProjects/edge-voice-classification)** - DS-CNN for ESP32 devices (+8% on noisy audio)

## Framework Integration
- **[PyTorch Lightning](./Examples/libraryExamples)** - Using PAI with popular training frameworks
- **[Hugging Face](./Examples/libraryExamples)** - Transformer models with dendritic layers

For framework-specific guidance, see our [Library Examples](./Examples/libraryExamples).

&nbsp;

# How It Works

Traditional artificial neurons are **point neurons** - they simply sum weighted inputs and apply an activation function. Real biological neurons have **dendrites** - branching structures that perform sophisticated computations before signals reach the cell body.

**Perforated AI** adds dendritic modules to artificial neurons:

1. **Automatic Perforation**: PAI analyzes your network and perforates it with dendrites for selected modules
2. **Validation-Driven Growth**: Identifies when to add dendrites based on validation score improvements
3. **Gradient-Free Growth**: Dendrites are added during training without requiring gradients through them
4. **Optimized Training**: Automatically adjusts learning rates for optimal dendritic optimization
5. **Multi-Layer Support**: Unlike other dendritic methods, PAI works on multiple neuron layers simultaneously

This approach delivers:
- ✅ **Significant accuracy improvements** on challenging tasks
- ✅ **Superior parameter efficiency** compared to standard scaling
- ✅ **Reduced data requirements** for achieving target performance
- ✅ **Robustness to noise** - particularly effective in real-world conditions
- ✅ **Drop-in compatibility** - works with existing PyTorch code

See our [Papers](./Papers) directory for detailed comparisons with other dendritic learning methods.

&nbsp;

# Results

Explore detailed performance benchmarks and real-world applications:

- **[Case Studies](https://www.perforatedai.com/case-studies)** - In-depth analysis of PAI across various domains and architectures
- **[January 2026 Hackathon Results](https://www.perforatedai.com/hackathon-results)** - Community-driven experiments and innovations with dendritic optimization

&nbsp;

# Python Version Support

We support Python 3.7+ and PyTorch 1.9+. We are committed to supporting Python versions for at least six months after their official end-of-life (EOL) date.

&nbsp;

# Contribution Guidelines

Perforated AI ❤️ open source! We welcome contributions from the community. 

**Adding Examples:**
- Follow the [MNIST example](./Examples/baseExamples/mnist) template
- Include before/after results with visualizations
- Provide complete running instructions

**Reporting Issues:**
- Visit [GitHub Issues](https://github.com/PerforatedAI/PerforatedAI/issues)
- Contact support@perforatedai.com

See the [Examples README](./Examples/README.md) for contribution guidelines.

&nbsp;

# Community

Join the Perforated AI community:
- 💬 [Discord](https://discord.gg/perforatedai) - Get help and share results
- 📧 [Newsletter](https://perforatedai.com/newsletter) - Stay updated on dendritic AI research
- 🐦 [Twitter](https://twitter.com/perforatedai) - Follow our latest developments

&nbsp;

# Citation

If you use Perforated AI in your research, please cite:

```bibtex
@article{brenner2025perforated,
  title={Perforated Backpropagation: A Neuroscience Inspired Extension to Artificial Neural Networks},
  author={Brenner, Ryan and others},
  journal={arXiv preprint arXiv:2501.18018},
  year={2025}
}
```

&nbsp;

# License

[Apache License 2.0](./LICENSE)

&nbsp;

# Alternative Training Mechanisms

When calling intitializePB a pb_neuron_layer_tracker called pai_tracker will be created.  This keeps track of all neuron modules and important values as well as performing the actions behind the scenes to add dendrite modules where they need to go.  It also must have a pointer to the optimizer being used. To get started quickly, or if the optimizer is hidden by a training framework, the following can be used:

    GPA.pai_tracker.set_optimizer_instance(optimizer)

However, we reccomend your optimizer and scheduler should be set this way instead. This method will automatically sweep over multiple learning rate options when adding dendrites, where often a lower learning rate is better for when after dendrites have been added. If you do use this method, the scheduler will get stepped inside our code so get rid of your scheduler.step() if you have one.  We recommend using ReduceLROnPlateau but any scheduler and optimizer should work.

    GPA.pai_tracker.set_optimizer(torch.optim.Adam)
    GPA.pai_tracker.set_scheduler(torch.optim.lr_scheduler.ReduceLROnPlateau)
    optimArgs = {'params':model.parameters(),'lr':learning_rate}
    schedArgs = {'mode':'max', 'patience': 5} #Make sure this is lower than epochs to switch
    optimizer, PAIscheduler = GPA.pai_tracker.setup_optimizer(model, optimArgs, schedArgs)
    
    Get rid of scheduler.step if there is one. If your scheduler is operating in a way
    that it is doing things in other functions other than just a scheduler.step this
    can cause problems and you should just not add the scheduler to our system.
    We leave this uncommented inside the code block so it is not forgotten.
    
    Another note - It seems that weight decay can sometimes cause problems with dendrite learning.  If you currently have weight decay and are not happy with the results, try without it.