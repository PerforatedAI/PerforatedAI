# 🚀 Quick Start Guide - Dendritic YOLOv8

Get running in 5 minutes!

## Option 1: Google Colab (Easiest - Recommended)

1. **Open the notebook**: [dendritic_yolov8_colab.ipynb](dendritic_yolov8_colab.ipynb)
2. **Upload to Colab**: File → Upload to Google Colab
3. **Select GPU**: Runtime → Change runtime type → T4 GPU
4. **Run all cells**: Runtime → Run all
5. **Wait ~2 hours** for training to complete
6. **Download results**: Check the last cell

Done! 🎉

## Option 2: Local Machine (Requires GPU)

```bash
# 1. Clone and install
git clone https://github.com/PerforatedAI/PerforatedAI.git
cd PerforatedAI
pip install setuptools
pip install -e .
cd Examples/hackathonProjects/dendritic-yolov8
pip install -r requirements.txt

# 2. Run quick test (baseline)
python train_yolov8_baseline.py --epochs 10

# 3. Run dendritic training
python train_yolov8_dendritic.py --epochs 50

# 4. Check results
ls PAI/PAI.png  # This is the required graph!
```

## Option 3: W&B Sweeps (Best Results)

```bash
# Login to W&B
wandb login

# Run sweep (will test many configurations)
python train_yolov8_dendritic.py \
    --use-wandb \
    --count 25 \
    --epochs 50

# View results at wandb.ai
```

## What You'll Get

After training completes, you'll have:

- ✅ `PAI/PAI.png` - **Required for hackathon submission**
- ✅ `PAI/PAIbest_test_scores.csv` - Detailed metrics
- ✅ Training curves and confusion matrices
- ✅ Trained model weights

## Next Steps

1. **Review `PAI/PAI.png`** - Make sure it looks good (dendrites should be visible)
2. **Calculate your metrics**:
   - Baseline mAP vs Dendritic mAP
   - Remaining Error Reduction (formula in README)
   - Parameter count comparison
3. **Update README.md** with your results
4. **Submit your PR** to the PerforatedAI repo

## Troubleshooting

**"CUDA out of memory"**
```bash
# Reduce batch size
python train_yolov8_dendritic.py --batch 8 --imgsz 320
```

**"No module named perforatedai"**
```bash
# Reinstall
cd /path/to/PerforatedAI
pip install -e .
```

**"PAI.png not generated"**
- Make sure training completed fully (don't interrupt!)
- Check `PAI/` folder exists
- Look for errors in console output

## Time Estimates

| Task | Time (T4 GPU) |
|------|---------------|
| Baseline (10 epochs) | ~5 minutes |
| Baseline (50 epochs) | ~15 minutes |
| Dendritic (50 epochs) | ~1-2 hours |
| W&B Sweep (25 runs) | ~24 hours |

## Tips for Hackathon Success

1. **Start with Colab** - Free GPU, no setup
2. **Run baseline first** - Get familiar with the pipeline
3. **Monitor PAI graph** - Should show multiple dendrite cycles
4. **Use W&B** - Extra points from judges!
5. **Document everything** - Screenshots, metrics, observations

Good luck! 🍀

Questions? Join the [PerforatedAI Discord](https://discord.gg/Fgw3FG3Hzt)
