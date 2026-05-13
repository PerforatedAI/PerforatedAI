<div align="center">

<img src="logo.png" width="400" alt="Perforated AI" />

### Better accuracy, smaller models, less data - enabled by perforated learning


<p>
<a href="https://pypi.python.org/pypi/perforatedai"><img src="https://img.shields.io/pypi/v/perforatedai" /></a>
<a href="https://pypi.python.org/pypi/perforatedai"><img src="https://img.shields.io/pypi/dm/perforatedai" /></a>
<a href="https://github.com/PerforatedAI/PerforatedAI"><img src="https://img.shields.io/github/stars/PerforatedAI/PerforatedAI" /></a>
<img src="https://img.shields.io/badge/python-3.7%2B-blue?logo=python" />
<img src="https://img.shields.io/badge/code%20style-black-000000.svg" />
<img src="https://img.shields.io/badge/PRs-welcome-brightgreen.svg" />
<a href="https://github.com/PerforatedAI/PerforatedAI/blob/main/LICENSE"><img src="https://img.shields.io/github/license/PerforatedAI/PerforatedAI" /></a>
</p>

</div>

---

## Introduction

Perforated AI (PAI) is a PyTorch library that adds artificial **dendrites** to your neural network - the same branching structures that give real biological neurons their computational power. PAI restructures your network *during* training and only requires minimal code changes.

## �� Key Results

- **Up to 70% accuracy improvements** on challenging tasks
- **Up to 90% compression** without sacrificing accuracy
- **50% reduction in data requirements** for similar accuracy scores

> These results come from the full **Perforated Backpropagation™** system, available in the [full Perforated AI Suite](#need-more-consider-the-full-perforated-ai-suite). [Case studies][case-studies] and our [published paper][arxiv] include full benchmark methodology.

## �� Quick Start

**Requirements:** Python 3.7+ · PyTorch 1.9+

```bash
pip install perforatedai
```

Integrate PAI into your training script with these five steps:

```python
from perforatedai import globals_perforatedai as GPA
from perforatedai import utils_perforatedai as UPA
import torch

model = YourModel()

# 1. Wrap your model after initialization
model = UPA.perforate_model(model)

# 2. Register your optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
GPA.pai_tracker.set_optimizer_instance(optimizer)

# 3. Run until dendritic optimization completes (replace your training loop with while True)
while True:
    train(model, optimizer)
    val_score = validate(model)

    # 4. Report validation score and handle model restructuring
    model, restructured, training_complete = GPA.pai_tracker.add_validation_score(val_score, model)
    model = model.to(device)  # re-send to device if PAI restructured the model

    # 5. React to PAI's outcome - exit when done, or reset optimizer after restructuring
    if training_complete:
        # Best model has been saved - training is done!
        break
    elif restructured:
        # Model was restructured with new dendrites - reinitialize optimizer (and scheduler) using your initial setup
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        GPA.pai_tracker.set_optimizer_instance(optimizer)
```

Need help with integration? See the [API documentation][api-docs] for full integration details, including how to use our [Claude skill][claude-skill] or [MCP server][https://github.com/PerforatedAI/PerforatedAI/blob/nn_customize/API/MCP_INSTALL.md] for AI coding assistants to get instant guidance.

## �� Examples

PAI works with many popular architectures and frameworks:

### Base Examples
- **[MNIST][mnist-example]** - Classic computer vision with dendritic enhancement
- **[TD3 Reinforcement Learning][td3-example]** - Continuous control with dendrites

### Advanced Examples
- **[ImageNet ResNet-18][imagenet-example]** - ResNet-18 with dendritic optimization ([pretrained model on Hugging Face][huggingface-model])
- **[Edge Impulse Block][edge-impulse-example]** - Keyword spotting in audio; across 800 hyperparameter sweeps, dendritic models were consistently more accurate at every parameter count

### Framework Integration
- **[PyTorch Lightning][lightning-example]** - Using PAI with popular training frameworks
- **[Hugging Face][hf-example]** - Transformer models with dendritic layers

## �� How It Works

Traditional artificial neurons are **point neurons** - they simply sum weighted inputs and apply an activation function. Real biological neurons have **dendrites** - branching structures that perform sophisticated computations before signals reach the cell body.

**Perforated AI** adds dendritic modules to artificial neurons:

1. **Automatic Perforation**: PAI analyzes your network and perforates it with dendrites for selected modules
2. **Validation-Driven Growth**: Identifies when to add dendrites based on validation score improvements
3. **Optimized Training**: Automatically adjusts learning rates for optimal dendritic optimization
4. **Multi-Layer Support**: Unlike other dendritic methods, PAI works on multiple neuron layers simultaneously

This approach delivers:
- **Significant accuracy improvements** on challenging tasks
- **Superior parameter efficiency** compared to standard scaling
- **Reduced data requirements** for achieving target performance
- **Robustness to noise** - particularly effective in real-world conditions
- **Drop-in compatibility** - works with existing PyTorch code

See our [Papers](./Papers) directory for detailed comparisons with other dendritic learning methods.

## �� Documentation

| Resource | Description |
|---|---|
| [API Documentation][api-docs] | Detailed integration instructions |
| [Customization Guide][customization-guide] | Advanced configuration options |
| [Output Guide][output-guide] | Understanding training visualizations |
| [Papers][papers] | Research comparisons and published work |

## �� Need more? Consider the full Perforated AI Suite

This repository is the open-source version of Perforated AI under Apache 2.0 - the dendritic architecture, trained with standard backpropagation. Free to use, fully self-serve.

For larger performance gains and a smoother setup experience, we also offer a commercial version that adds:
- **Perforated Backpropagation™** - our patented training algorithm (replaces standard backprop) that produces our headline accuracy, compression, and data-efficiency results
- **Perforated Studio** - a GUI for configuring runs and inspecting results, so you can get up and running without hand-rolling the integration in code

[Get in touch][get-started] to learn more about the commercial version.

## �� Contributing

We welcome all contributions from the community!

**Adding Examples:**
- Follow the [MNIST example][mnist-example] template
- Include before/after results with visualizations
- Provide complete running instructions
- Provide baseline code as `NAME.py` and perforated code as `NAME_perforated.py`

**Reporting Issues:**
- [GitHub Issues][github-issues]
- support@perforatedai.com

**Modifying Code:**
- Describe what the change does and the benefit it provides
- Use Black Python formatting
- Include comments within code to describe processes

## �� Community

- �� [Discord][discord] - Get help and share results
- �� [Newsletter][newsletter] - Stay updated on dendritic AI research
- �� [LinkedIn][linkedin] - Follow our latest developments
- �� [Hugging Face][huggingface] - Try our pretrained models on your datasets

## Citation

If you use Perforated AI in your research, please cite:

```bibtex
@article{brenner2025perforated,
  title={Perforated Backpropagation: A Neuroscience Inspired Extension to Artificial Neural Networks},
  author={Brenner, Rorry and Itti, Laurent},
  journal={arXiv preprint arXiv:2501.18018},
  year={2025}
}
```

## Python Version Support

We support Python 3.7+ and PyTorch 1.9+. We are committed to supporting Python versions for at least six months after their official end-of-life (EOL) date.


## License

[Apache License 2.0][license-url]

---

[pypi-url]: https://pypi.python.org/pypi/perforatedai
[github-url]: https://github.com/PerforatedAI/PerforatedAI
[license-url]: ./LICENSE
[api-docs]: ./API/README.md
[claude-skill]: .github/skills/perforatedai/SKILL.md
[customization-guide]: ./API/customization.md
[output-guide]: ./API/output.md
[papers]: ./Papers
[case-studies]: https://www.perforatedai.com/case-studies
[arxiv]: https://arxiv.org/pdf/2501.18018
[mnist-example]: ./Examples/baseExamples/mnist
[td3-example]: https://github.com/PerforatedAI/PerforatedAI/tree/develop/Examples/reinforcementLearning/td3_cheetah
[imagenet-example]: ./Examples/imagenet
[huggingface-model]: https://huggingface.co/perforated-ai/resnet-18-perforated-gd
[edge-impulse-example]: ./Examples/hackathonProjects/example-custom-ml-block-pytorch
[lightning-example]: ./Examples/libraryExamples/pytorch_lightning
[hf-example]: ./Examples/libraryExamples/huggingface
[get-started]: https://www.perforatedai.com/get-started
[github-issues]: https://github.com/PerforatedAI/PerforatedAI/issues
[discord]: https://discord.gg/Fgw3FG3Hzt
[newsletter]: https://www.perforatedai.com/contact
[linkedin]: https://www.linkedin.com/company/perforated-ai
[huggingface]: https://huggingface.co/perforated-ai
