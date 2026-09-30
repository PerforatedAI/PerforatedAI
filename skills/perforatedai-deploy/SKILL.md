---
name: perforatedai-deploy
description: "Deploy a trained PerforatedAI model: load a _pai checkpoint for inference or fine-tuning, and export to ONNX, TFLite, or TorchScript for non-PyTorch platforms. Triggers: 'load my perforated model for inference', 'export my perforated model', 'convert my perforated model to ONNX/TFLite/TorchScript', 'deploy my dendritic model'. For adding dendrites to a model in the first place, use the perforatedai skill."
---

# PerforatedAI Deployment

Everything after training: getting a trained dendritic model into production.

**Two entry points:**

- **"Load my perforated model for inference"** - load a `_pai` checkpoint in PyTorch, run inference, or fine-tune it further. Recipients need `perforatedai` installed but **not** `perforatedbp`.
- **"Export my perforated model"** - convert to ONNX, TFLite, or TorchScript for platforms without PyTorch. The only PAI-specific requirement is a cleanup call before export; the rest is standard.

This skill assumes training is already done. To add dendrites to a model, use the **perforatedai** skill. To debug a training integration, use **perforatedai-debugging**.

---

## Loading a Trained Model for Inference or Fine-tuning

When the user wants to load a trained perforated model for **inference** or **deployment** (not continued training), use this workflow.

**Important: This is for loading models WITHOUT the PAI tracker for inference/deployment. For resuming training or transfer learning, see [api/customization.md Section 7](https://github.com/PerforatedAI/PerforatedAI/blob/main/api/customization.md#7-loading).**

---

**Step 1: Confirm the use case**

Ask: "Are you loading this model for:
1. **Inference/deployment** (make predictions, no training)
2. **Fine-tuning** (standard PyTorch training with frozen dendrites)
3. **Continued dendrite training** (resume PAI training with more dendrite additions)
4. **Transfer learning** (retrain for new task with dendrite additions)

Options 3 and 4 require different loading functions - see [api/customization.md Section 7](https://github.com/PerforatedAI/PerforatedAI/blob/main/api/customization.md#7-loading)."

**If they choose option 1 or 2** (inference or fine-tuning without dendrite additions), proceed with steps below.

**If they choose option 3 or 4**, tell them:
> "For continued dendrite training or transfer learning, you need `UPA.load_system()` or `UPA.load_pretrained_model()` instead. See the Loading section in [api/customization.md](https://github.com/PerforatedAI/PerforatedAI/blob/main/api/customization.md#7-loading) for details. I can help you implement that if needed."

---

**Step 2: Verify _pai checkpoint exists**

**First, check if they enabled _pai saves during training:**

Ask: "During training, did you set `GPA.pc.set_pai_saves(True)` in your configuration?"

**If YES or they're not sure:**
- Tell them: "If you set `set_pai_saves(True)`, PAI automatically created optimized `_pai` checkpoints. Let me check for them."
- Ask: "What was your `save_name` during training? (default is 'PAI')"
- Check if files like `{save_name}/best_model_pai.pt` or `{save_name}/latest_pai.pt` exist

**If they have _pai checkpoints:**
- Proceed to Step 3 with `load_pai_model()`

**If NO _pai checkpoints found:**
- Tell them: "I don't see any `_pai` checkpoints. You can still load regular checkpoints, but they include training scaffolding which increases memory usage. For production deployment, I recommend re-running a short training session with `GPA.pc.set_pai_saves(True)` to generate optimized checkpoints."
- Ask: "Would you like to:
  1. Load the regular checkpoint anyway (works but less efficient)
  2. Re-run training briefly with `set_pai_saves(True)` to generate optimized checkpoints"

**If they choose option 1**, proceed to Step 3 but note the limitation in your explanation.

---

**Step 3: Create inference script**

Create a new Python script (or ask where they want the code added) with the following pattern:

```python
from perforatedai import network_perforatedai as NPA
import torch

# Step 1: Create the base model architecture (same as during training)
model = YourModelClass()  # Use same architecture as training

# Step 2: Load the _pai checkpoint
# This automatically:
# - Calls convert_network() to wrap modules
# - Reconstructs dendrite structure
# - Loads all weights (neurons + dendrites)
model = NPA.load_pai_model(model, 'PAI/best_model_pai.pt')

# Step 3: Set to eval mode and move to device
model.eval()
model = model.to('cuda')  # or 'cpu'

# Step 4: Run inference
with torch.no_grad():
    output = model(input_data)
    # Process output as needed
```

**Key points to explain:**

1. **No PAI tracker needed**: `load_pai_model()` handles everything - no `perforate_model()`, no `GPA.pc` configuration, no tracker initialization
2. **Model architecture must match**: They need to instantiate the same base model class used during training
3. **Dendrites are frozen**: The loaded model has dendrites integrated but frozen - perfect for inference
4. **No training scaffolding**: `_pai` checkpoints have minimal memory footprint

---

**Step 4: Fine-tuning (optional)**

If they chose fine-tuning in Step 1, add this after loading:

```python
# Load the model (same as above)
model = NPA.load_pai_model(model, 'PAI/best_model_pai.pt')
model = model.to('cuda')

# Standard PyTorch fine-tuning (dendrites stay frozen)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(num_epochs):
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        output = model(batch)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
    
    # Save fine-tuned model
    torch.save(model.state_dict(), 'finetuned_model.pt')
```

Explain: "This fine-tunes the entire model (neurons + frozen dendrites) using standard PyTorch. The dendrite structure remains fixed - no new dendrites are added."

---

**Step 5: Sharing/Deployment**

If they're deploying or sharing the model, provide additional guidance:

**For sharing with others:**
```python
# Recipients need perforatedai installed but NOT perforatedbp
# pip install perforatedai

# Then they can load with the same code:
model = YourModelClass()
model = NPA.load_pai_model(model, 'best_model_pai.pt')
```

**For production deployment:**
- `_pai` models are self-contained - just ship the `.pt` file and model architecture definition
- No PAI tracker or configuration needed
- Dendrites are baked into the model structure

---

**Common questions:**

**Q: Can I convert a regular checkpoint to _pai format?**
A: No, you need to enable `GPA.pc.set_pai_saves(True)` during training. If you only have regular checkpoints, you can:
1. Load with `UPA.load_system()` (requires full PAI setup)
2. Run a few training epochs with `set_pai_saves(True)` to generate `_pai` versions

**Q: What's the difference between `load_pai_model()` and `load_system()`?**
A: 
- `load_pai_model()`: Inference/deployment, no tracker, dendrites frozen, minimal memory
- `load_system()`: Resume training, requires tracker, can add more dendrites, keeps all training state

**Q: Can I add more dendrites after loading with `load_pai_model()`?**
A: No. For continued dendrite training, use `UPA.load_system()` or `UPA.load_pretrained_model()` instead. See [api/customization.md Section 7](https://github.com/PerforatedAI/PerforatedAI/blob/main/api/customization.md#7-loading).

---

## Exporting to ONNX / TFLite / TorchScript

When the user wants to export a trained perforated model to **ONNX**, **TFLite**, or **TorchScript** for deployment on non-PyTorch platforms (mobile, edge devices, web, C++ runtimes), use this workflow.

**Important: This is for exporting models to platform-agnostic formats AFTER training completes. For PyTorch inference, see entry point #4 ("Load my perforated model for inference").**

---

**Step 1: Confirm the export target**

Ask: "What format do you need to export to?
1. **ONNX** (most platforms, TensorRT, ONNX Runtime)
2. **TFLite** (Android, iOS, embedded systems via TensorFlow Lite)
3. **TorchScript** (C++ production, PyTorch Serve)
4. **Other** (specify your target platform)"

**Common use cases:**
- **ONNX**: Cross-platform inference, GPU acceleration with TensorRT, ONNX Runtime deployment
- **TFLite**: Mobile apps (Android/iOS), microcontrollers, edge TPUs
- **TorchScript**: Production C++ environments, PyTorch Serve, when you need full PyTorch compatibility

---

**Step 2: Apply PAI cleanup (REQUIRED for all export formats)**

**CRITICAL: You must apply these two PAI transformations BEFORE exporting. They convert PAI's dendritic wrapper structure into standard PyTorch operations.**

```python
from perforatedai import blockwise_perforatedai as BPA
from perforatedai import clean_perforatedai as CPA

# Set to eval mode (standard PyTorch requirement)
model.eval()

# Apply PAI cleanup - THE ONLY PAI-SPECIFIC REQUIREMENT
model = BPA.blockwise_network(model)
model = CPA.refresh_net(model)
```

**What these do:**
- `BPA.blockwise_network()`: Converts dendritic computation into standard PyTorch Conv/Linear blocks
- `CPA.refresh_net()`: Removes PAI wrapper classes (`PAINeuronModule`, `PAIDendriteModule`), leaving pure PyTorch modules

**Why this is required:**
- PAI wrapper classes are training scaffolding that ONNX/TFLite/TorchScript exporters don't understand
- After cleanup, dendrites become standard Conv/Mul/Add operations that trace cleanly
- Cleanup is one-way: you can't resume PAI training after this (save a checkpoint before cleanup if needed)

**Optional (not PAI-specific):**
- `model.cpu()` - Moves to CPU before export (common practice, but can export on GPU)
- Output suppression - BPA/CPA print some info; suppress with `contextlib.redirect_stdout()` if desired

---

**Step 3: Export to ONNX**

After applying BPA + CPA cleanup:

```python
import torch

# Prepare dummy input matching your model's input shape
# Example: for (batch, channels, height, width)
dummy_input = torch.randn(1, 3, 224, 224)  # Adjust shape to match your model

onnx_path = 'model.onnx'

# Standard ONNX export
torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    export_params=True,
    opset_version=11,  # 10+ works; higher versions have more operators
    do_constant_folding=True,  # Optimize constants at export time
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},  # Optional: allow variable batch size
    dynamo=False  # Optional: use legacy exporter for better compatibility
)

print(f"Exported to {onnx_path}")
```

**Key export parameters explained:**
- `opset_version=11`: Use 10+ for modern operators; higher is fine, not PAI-specific
- `do_constant_folding=True`: Bakes constants at export time, reduces runtime computation
- `dynamic_axes`: Optional - allows variable batch size; omit for fixed batch (sometimes faster)
- `dynamo=False`: Optional - uses legacy `torch.onnx.export`; omit or set to `True` for newer `torch.export` path

**Verify the export:**
```python
import onnxruntime as ort

# Load and test the ONNX model
ort_session = ort.InferenceSession(onnx_path)
test_input = dummy_input.numpy()
ort_outputs = ort_session.run(None, {ort_session.get_inputs()[0].name: test_input})
print(f"ONNX output shape: {ort_outputs[0].shape}")
```

---

**Step 4: Convert ONNX to TFLite (optional, for mobile/edge)**

If the user needs TFLite (for Android, iOS, microcontrollers):

**Install conversion tools:**
```bash
pip install onnx onnxsim onnx2tf tensorflow
```

**Convert:**
```python
import subprocess
import sys
import onnx
import onnxsim

# Step 1: Simplify ONNX (RECOMMENDED for TFLite, not PAI-specific)
# This is a workaround for onnx2tf/TFLite issues with quantization.
# It folds standalone Constant ops into consuming ops, preventing
# INT8 quantization failures. Skip this if you only need float32 TFLite.
onnx_model = onnx.load(onnx_path)
simplified_model, check = onnxsim.simplify(onnx_model)
simplified_path = 'model_simplified.onnx'
onnx.save(simplified_model, simplified_path)
print("✓ Simplified ONNX model")

# Step 2: Convert to TensorFlow SavedModel
saved_model_dir = 'saved_model'
result = subprocess.run(
    [sys.executable, "-m", "onnx2tf", "-i", simplified_path, "-o", saved_model_dir, "-osd"],
    capture_output=True, text=True, timeout=120
)
if result.returncode != 0:
    print(f"onnx2tf error: {result.stderr}")
    raise Exception("onnx2tf conversion failed")
print(f"✓ Converted to TF SavedModel at {saved_model_dir}")

# Step 3: Convert to TFLite
import tensorflow as tf

# Float32 TFLite (no quantization)
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
tflite_model = converter.convert()
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)
print("✓ Created float32 TFLite model")

# Optional: INT8 quantization (smaller, faster, slight accuracy loss)
# Requires representative dataset for calibration
def representative_dataset():
    # Provide 100-500 samples from your training data
    for i in range(100):
        sample = ...  # Get a sample from your dataset
        yield [sample.astype(np.float32)]

converter_quant = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter_quant.optimizations = [tf.lite.Optimize.DEFAULT]
converter_quant.representative_dataset = representative_dataset
tflite_quant_model = converter_quant.convert()
with open('model_int8.tflite', 'wb') as f:
    f.write(tflite_quant_model)
print("✓ Created INT8 quantized TFLite model")
```

**TFLite conversion notes:**
- **onnxsim is optional** (only needed to fix INT8 quantization issues with onnx2tf - NOT a PAI requirement)
- Float32 TFLite: ~same size as ONNX, full accuracy, no simplification needed
- INT8 quantized: ~4× smaller, faster on mobile/edge, slight accuracy drop (<1% typical)
- For INT8, provide representative samples from your actual data distribution
- These are **TFLite/onnx2tf workarounds**, not PAI-specific requirements

---

**Step 5: Export to TorchScript (optional, for C++ deployment)**

If the user needs TorchScript (for C++ environments, PyTorch Serve):

```python
# After BPA + CPA cleanup
model.eval()

# Method 1: Tracing (recommended, usually works)
example_input = torch.randn(1, 3, 224, 224)
traced_model = torch.jit.trace(model, example_input)
traced_model.save('model_traced.pt')
print("✓ Exported TorchScript via tracing")

# Method 2: Scripting (if tracing fails due to control flow)
# Use this if your model has if/for statements in forward()
try:
    scripted_model = torch.jit.script(model)
    scripted_model.save('model_scripted.pt')
    print("✓ Exported TorchScript via scripting")
except Exception as e:
    print(f"Scripting failed: {e}")
    print("Use tracing instead (Method 1)")
```

**Load in C++:**
```cpp
#include <torch/script.h>
torch::jit::script::Module module = torch::jit::load("model_traced.pt");
```

---

**Common Issues and Solutions**

**Issue: "ONNX export fails with 'symbolic not implemented'"**
- **PAI-related?** No - general PyTorch/ONNX issue
- **Cause**: Your model uses an operator that doesn't have ONNX symbolic mapping
- **Fix**: Check which layer is failing. Sometimes custom operations or very new PyTorch ops don't have ONNX support
- **Workaround**: Rewrite the operation using supported ops, or use TorchScript instead

**Issue: "TFLite quantization shows fully_quantize: 0"**
- **PAI-related?** No - onnx2tf/TFLite issue
- **Cause**: Model has standalone Constant tensors that onnx2tf generates
- **Fix**: Run `onnxsim.simplify()` before `onnx2tf` to fold constants into consuming ops
- **Verification**: Look for "fully_quantize: 1" in the conversion output

**Issue: "MaxPool with ceil_mode produces PartitionedCall in TF"**
- **PAI-related?** No - PyTorch to TFLite conversion issue
- **Cause**: PyTorch `MaxPool(ceil_mode=True)` creates ops that TFLite can't trace through
- **Fix**: Pad your input to even dimensions before MaxPool, then use `ceil_mode=False` and `stride=2`
- **Example**: See lines 489-494 in [train.py](https://github.com/PerforatedAI/PerforatedAI/blob/main/examples/submitted_projects/perforated-impulse-nn-block/train.py)

**Issue: "Exported model is much larger than PyTorch checkpoint"**
- **PAI-related?** No - normal for all ONNX/TFLite exports
- **Cause**: Normal - ONNX/TFLite store the full compute graph, not just weights
- **Solution**: For smallest size, use INT8 quantized TFLite (typically 4× compression)

**Issue: "Accuracy drops after export"**
- **PAI-related?** Possibly - verify BPA/CPA were applied correctly
- **Cause**: Usually quantization (INT8), sometimes numerical precision differences, rarely incomplete cleanup
- **Fix**: 
  - **First**: Verify BPA + CPA were applied (check model output matches before/after cleanup with same input)
  - Compare float32 TFLite vs INT8 - if INT8 is the problem, adjust quantization or use float16
  - For ONNX, try `opset_version=11` or higher

**Issue: "Model with BatchNorm fails to export"**
- **PAI-related?** No - general PyTorch export issue
- **Cause**: BatchNorm in eval mode should fold into previous conv layer but sometimes doesn't
- **Fix**: Call `model.eval()` before BPA/CPA cleanup. The cleanup folds BatchNorm automatically

---

**Reference Implementation**

For a complete working example that exports to ONNX and TFLite (including INT8 quantization), see:
- [examples/submitted_projects/perforated-impulse-nn-block/train.py](https://github.com/PerforatedAI/PerforatedAI/blob/main/examples/submitted_projects/perforated-impulse-nn-block/train.py) (lines 1006-1220)

**What's PAI-specific in this example:**
- BPA + CPA cleanup (lines 1006-1011) - **REQUIRED**

**What's Edge Impulse/TFLite-specific (NOT PAI requirements):**
- Output suppression - just keeps console clean
- ONNX simplification (lines 1188-1192) - workaround for onnx2tf INT8 quantization
- MaxPool padding tricks (lines 489-494, 523-529) - workaround for TFLite PartitionedCall issue
- Dynamic batch size handling - Edge Impulse requirement
- INT8 quantization with calibration - deployment optimization choice
- ONNX simplification (lines 1188-1192)
- TFLite conversion with INT8 quantization (lines 1195-1220)
- Handling Edge Impulse deployment requirements

---

**Summary Checklist**

**PAI-specific requirements (MUST do):**
- ✅ Applied `BPA.blockwise_network()` 
- ✅ Applied `CPA.refresh_net()`
- ✅ Model in eval mode (`model.eval()`)

**Standard PyTorch export practices (recommended):**
- ✅ Model on CPU (`model.cpu()`) - optional, can export on GPU
- ✅ Created dummy input with correct shape
- ✅ Tested model output matches before/after cleanup (optional but helpful for debugging)

**For ONNX:**
- ✅ Set `opset_version=11` or higher (10 works too, just older)
- ✅ Set `dynamo=False` to use legacy exporter (better compatibility)
- ✅ Verified export with ONNX Runtime

**For TFLite (onnx2tf workarounds, not PAI-specific):**
- ✅ Ran `onnxsim.simplify()` on ONNX model (ONLY if doing INT8 quantization)
- ✅ Converted with `onnx2tf`
- ✅ Created float32 TFLite baseline (no simplification needed)
- ✅ (Optional) Created INT8 quantized version with representative dataset

**For TorchScript:**
- ✅ Used `torch.jit.trace()` (or `script()` if tracing fails)
- ✅ Saved `.pt` file for C++ loading

---

