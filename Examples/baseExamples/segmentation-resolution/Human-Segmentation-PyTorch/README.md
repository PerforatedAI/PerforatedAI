# Human Segmentation Resolution Experiments

This repository records a resolution-scaling experiment for human segmentation with a UNet model, along with PerforatedAI runs that recover much of the accuracy lost at lower input resolutions.

It is based on the original Human-Segmentation-PyTorch project from AntiAegis:

- Original repository: https://github.com/AntiAegis/Human-Segmentation-PyTorch
- Upstream segmentation project page: https://github.com/AntiAegis/Semantic-Segmentation-PyTorch

The results summarized here come from [results.txt](/home/rbrenner/PAI/PerforatedAI/Examples/baseExamples/segmentation-resolution/Human-Segmentation-PyTorch/results.txt).

## Experiment Summary

The experiment was run in four stages:

1. Run the original UNet training setup at the default image resolution.
2. Reduce the input image resolution to half and quarter and observe the drop in segmentation quality.
3. Run the PerforatedAI-enabled training code at the same reduced resolutions.
4. Compare the lower-resolution baselines against the perforated runs to measure how much of the accuracy deficit is recovered.

The command used for the reduced-resolution experiments is:

```bash
CUDA_VISIBLE_DEVICES=1 ENV/bin/python train.py --config config/config_UNet.json --device 0 --resolution-scale half
```

Use `half` or `quarter` for the `--resolution-scale` argument or no value for the default resolution.

## Dataset Used

The command above uses [config/config_UNet.json](/home/rbrenner/PAI/PerforatedAI/Examples/baseExamples/segmentation-resolution/Human-Segmentation-PyTorch/config/config_UNet.json), and that config points to these pair files:

- `dataset/train_supervisely.txt`
- `dataset/valid_supervisely.txt`

Data download link:

- https://drive.google.com/file/d/1Y1atvePuMx1pyIOVJNGgJ_jVNBy_Bds8/view?usp=drive_link

So this experiment is configured for the Supervisely portrait-person split used by the UNet configuration.

## Recorded Results

The main comparison metric in [results.txt](/home/rbrenner/PAI/PerforatedAI/Examples/baseExamples/segmentation-resolution/Human-Segmentation-PyTorch/results.txt) is validation mIoU.

| Run | Resolution | PerforatedAI | Valid mIoU | Valid loss |
| --- | --- | --- | ---: | ---: |
| Original | full | no | 0.8420 | 0.0978 |
| Reduced | half | no | 0.8204 | 0.1124 |
| Reduced | quarter | no | 0.7699 | 0.1465 |
| Perforated | half | yes | 0.8756 | 0.0748 |
| Perforated | quarter | yes | 0.8241 | 0.1087 |

## Interpretation

- Reducing the input resolution from full to half lowers validation mIoU from `0.8420` to `0.8204`.
- Reducing the input resolution from full to quarter lowers validation mIoU more sharply, to `0.7699`.
- Adding PerforatedAI at half resolution raises validation mIoU to `0.8756`, which is higher than the recorded full-resolution baseline.
- Adding PerforatedAI at quarter resolution raises validation mIoU to `0.8241`, recovering most of the loss introduced by quarter-resolution inputs.

In short, lowering resolution hurts segmentation accuracy, and the perforated model substantially recovers that lost accuracy at both reduced resolutions.

## How To Reproduce

### 1. Original baseline

The original scores should be reproduced by running the corresponding code in the original upstream Human-Segmentation-PyTorch repository with the same UNet configuration.

Upstream references:

- Original repository: https://github.com/AntiAegis/Human-Segmentation-PyTorch
- Original training and benchmark documentation: https://github.com/AntiAegis/Semantic-Segmentation-PyTorch

If you want the non-PerforatedAI code path from this workspace, [train_original.py](/home/rbrenner/PAI/PerforatedAI/Examples/baseExamples/segmentation-resolution/Human-Segmentation-PyTorch/train_original.py) mirrors that baseline behavior and also supports `--resolution-scale`.

Example:

```bash
CUDA_VISIBLE_DEVICES=1 ENV/bin/python train_original.py --config config/config_UNet.json --device 0
```

### 2. Reduced-resolution scores without dendrites

To measure the effect of resolution reduction alone in this modified codebase, run the resolution-scaled training with dendrites disabled by setting:

```python
GPA.pc.set_max_dendrites(0)
```

Add that to the PerforatedAI configuration in [train.py](/home/rbrenner/PAI/PerforatedAI/Examples/baseExamples/segmentation-resolution/Human-Segmentation-PyTorch/train.py) so the model runs without any dendrites, then launch:

```bash
CUDA_VISIBLE_DEVICES=1 ENV/bin/python train.py --config config/config_UNet.json --device 0 --resolution-scale half
CUDA_VISIBLE_DEVICES=1 ENV/bin/python train.py --config config/config_UNet.json --device 0 --resolution-scale quarter
```

These runs isolate the effect of using smaller input images in the resolution-modified code.

### 3. Perforated runs

Run the same commands without forcing the dendrite count to zero:

```bash
CUDA_VISIBLE_DEVICES=1 ENV/bin/python train.py --config config/config_UNet.json --device 0 --resolution-scale half
CUDA_VISIBLE_DEVICES=1 ENV/bin/python train.py --config config/config_UNet.json --device 0 --resolution-scale quarter
```

Those runs enable the PerforatedAI path defined in [train.py](/home/rbrenner/PAI/PerforatedAI/Examples/baseExamples/segmentation-resolution/Human-Segmentation-PyTorch/train.py).

## Relevant Files

- [results.txt](/home/rbrenner/PAI/PerforatedAI/Examples/baseExamples/segmentation-resolution/Human-Segmentation-PyTorch/results.txt): recorded training and validation metrics
- [config/config_UNet.json](/home/rbrenner/PAI/PerforatedAI/Examples/baseExamples/segmentation-resolution/Human-Segmentation-PyTorch/config/config_UNet.json): UNet experiment configuration
- [train.py](/home/rbrenner/PAI/PerforatedAI/Examples/baseExamples/segmentation-resolution/Human-Segmentation-PyTorch/train.py): PerforatedAI-enabled training entry point
- [train_original.py](/home/rbrenner/PAI/PerforatedAI/Examples/baseExamples/segmentation-resolution/Human-Segmentation-PyTorch/train_original.py): baseline training entry point without PerforatedAI
