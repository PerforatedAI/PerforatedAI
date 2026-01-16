# MobileNetV2 Audio Classification - THE PERFECT HACKATHON SUBMISSION

**MobileNet is the IDEAL choice for this hackathon** - it hits every scoring criterion perfectly.

## Why MobileNetV2 is Perfect

### ✅ **High Project Prevalence**
- MobileNet is **one of the most widely deployed models in production**
- Used by Google, Facebook, mobile apps worldwide
- Standard benchmark for efficient neural networks
- Rubric says: "broader prevalence will be scored more favorably"

### ✅ **Pretrained & Transfer Learning**
- Pretrained on ImageNet (1.2M images)
- Adapted for audio via mel-spectrograms
- Shows dendrites enhance transfer learning

### ✅ **Fast Training (~2 hours total)**
- Only ~3.5M parameters (vs AST's 86M)
- Trains quickly on MacBook Pro
- Batch size 32 (vs AST's 8)
- **Perfect for your timeline**

### ✅ **"Mobile" = Edge Deployment Story**
- Literally designed for mobile devices
- Your narrative writes itself
- Built-in credibility for edge deployment

## Model Comparison

| Model | Params | Training Time | Expected Baseline | Expected with PAI | Prevalence | Edge Story |
|-------|--------|---------------|-------------------|-------------------|------------|------------|
| CNN14 (scratch) | 1.6M | 2h | 80-85% | 83-88% | Low | Good |
| AST (transformer) | 86M | 6h+ | 90-95% | 92-96% | High | Poor |
| **MobileNetV2** | **3.5M** | **2h** | **85-90%** | **88-92%** | **Very High** | **Perfect** |

## Expected Results

**Baseline MobileNetV2 (ImageNet → UrbanSound8K):**
- Test Accuracy: **85-90%**
- Better than from-scratch CNN14 (80-85%)
- Competitive with transformers but 25x smaller

**MobileNetV2 + PerforatedAI:**
- Test Accuracy: **88-92%**
- 3-5% improvement
- Significant error reduction
- Still edge-deployable

## Your Winning Narrative

### **"Dendritic Optimization of Production-Scale Mobile Models"**

**Key Points:**
1. **MobileNet is industry-standard** for edge AI
2. **Used in billions of mobile devices** worldwide
3. **Pretrained transfer learning** (modern ML practice)
4. **Dendrites make it even better** for deployment
5. **Perfect balance**: High accuracy + Low compute

**Addresses rubric directly:**
- ✅ High prevalence (MobileNet is everywhere)
- ✅ Quality optimization (3-5% improvement + error reduction)
- ✅ Clear narrative (mobile/edge deployment)
- ✅ Business impact (production-scale model)

## Files Created

1. **`utils/mobilenet_audio_model.py`**: MobileNetV2 adapted for audio
2. **`train_baseline_mobilenet.py`**: Baseline training
3. **`train_perforatedai_mobilenet.py`**: Training with dendrites

## Usage

**You've already preprocessed UrbanSound8K!** Just run:

### Step 1: Train Baseline (45-60 min)

```bash
python train_baseline_mobilenet.py
```

### Step 2: Train with Dendrites (60-90 min)

```bash
python train_perforatedai_mobilenet.py
```

**Total: ~2 hours** (vs AST's 6+ hours)

## Technical Details

**Architecture:**
- Base: MobileNetV2 (ImageNet pretrained)
- Modified: First conv layer (RGB → Grayscale mel-spectrogram)
- Modified: Final classifier (1000 classes → 10 classes)
- Inverted residuals + depthwise separable convolutions

**Training Strategy:**
- Transfer learning from ImageNet
- Fine-tune on mel-spectrograms (128 x 173)
- Lower learning rate (0.001 → 0.0001 after improvements)
- ReduceLROnPlateau scheduler

**Why This Works:**
- Mel-spectrograms are 2D (like images)
- Pretrained features transfer well to audio
- MobileNet's efficiency architecture helps with audio too

## Real-World Applications

**MobileNet + Audio = Perfect for:**
- **Smart speakers** (Google Home, Alexa)
- **Mobile apps** (sound recognition)
- **Hearing aids** (ultra-low power)
- **IoT sensors** (environmental monitoring)
- **Automotive** (in-car audio detection)

**With Dendrites:**
- Better accuracy without sacrificing efficiency
- Still fits on mobile devices
- Lower latency, lower power consumption
- Demonstrates production-viable optimization

## Comparison to SOTA

**HTSAT-22 (Transformer SOTA):**
- 30M parameters, 98% accuracy
- Too large for mobile devices

**MobileNetV2 + Dendrites:**
- 3.5M parameters, 88-92% accuracy
- **8.6x smaller** than SOTA
- Deployable on smartphones, IoT
- "Good enough" accuracy for most applications

**Your trade-off is defensible:**
- Sacrifice 6-10% accuracy
- Gain 8.6x size reduction
- Enable mobile/edge deployment
- Maintain fast inference

## Time Breakdown

| Task | Time |
|------|------|
| Preprocessing | Already done! ✓ |
| Baseline training | 45-60 min |
| PAI training | 60-90 min |
| **Total** | **~2 hours** |

## Why This Wins the Hackathon

**Your submission:**
> "Dendritic Optimization of MobileNetV2 for Mobile Audio Classification"

**Why judges will love it:**
1. **High prevalence** - MobileNet is industry-standard
2. **Practical** - Production-scale model, not toy CNN
3. **Fast** - Reproducible results in 2 hours
4. **Meaningful** - 88-92% accuracy is deployment-viable
5. **Clear impact** - Mobile devices, edge AI, IoT
6. **Modern ML** - Transfer learning + dendritic optimization

**This checks every box in the rubric while being achievable TODAY.**

## Start Now!

```bash
python train_baseline_mobilenet.py
```

While it trains, write your submission narrative around:
- MobileNet's ubiquity in production
- Edge AI deployment
- Real-world cost/latency benefits
- Dendrites making mobile models even better

**This is your winning submission.** 🎯
