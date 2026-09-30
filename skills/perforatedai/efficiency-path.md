# Efficiency Path - Downsizing Before Adding Dendrites

Reference file for the **perforatedai** skill, Step 1.3. Read this only when the user says they're
optimizing for **efficiency / model size**. It does not apply to the accuracy path.

The goal: start from a *smaller* base model than they're using now, then add dendrites to recover
the original model's performance at a lower parameter count. Dendrites add parameters, so
"efficiency" means choosing a smaller starting architecture, not perforating the existing one.

When you finish here, return to Step 1.4 of the main skill to record the baseline.

---

- **Based on your earlier analysis of their training script, determine if model selection is configurable:**
  - Does it have command-line arguments that specify model architecture, variant, or size (e.g., `--model`, `--arch`, `--model_name`, `--resnet_version`, `--width`, `--depth`, `--num_heads`, `--num_layers`, `--hidden_dim`)?
  - Does it have config files that specify model architecture or size parameters?
  - Or is the model hardcoded in the script?

- **Then proceed based on what you find:**

  **If model IS configurable (has --model, --arch, or config arguments):**
  - **ASK what they're currently using, tailored to what's configurable:**
    - If they have `--model` or `--arch` argument: "I see your script has a `--model` argument. What model are you currently using?"
    - If they have architecture size parameters (e.g., `--width`, `--depth`, `--num_heads`, `--num_layers`): "I see your script supports configurable architecture. What are your current settings? (e.g., width, depth, number of heads/layers)"
    - If they have both model selection AND size parameters: "I see your script supports different configurations. What model and settings are you currently using? (e.g., model name, width, depth, number of layers/heads)"
  - Wait for their answer before proceeding
  
  **If model is NOT configurable (hardcoded in script):**
  - **Analyze the code** to identify what model they're using
  - Tell them: "I can see you're using [ModelName] in your script."
  - Proceed directly based on what you found (no need to ask)

- **Then determine the path based on their model (🚨 CRITICAL: For efficiency, the model MUST be smaller - always recommend downsizing):**

  **For Pretrained Models (e.g., ResNet50, BERT-base, GPT2-medium):**
  
  **If they're using a larger variant within a model family (ResNet50, ResNet34, BERT-base, GPT2-medium, etc.):**
  - Tell them: "You're using [ModelName]. To optimize for efficiency, I have two options for you.
    
    Important: Dendritic optimization often allows a smaller architecture to achieve the accuracy goals you would have previously needed the larger model for. The dendrites add targeted capacity exactly where needed, making efficient architectures surprisingly powerful.
    
    **Option 1: Smaller variant in the same family**
    - ResNet50 → ResNet18 + dendrites (or ResNet34 as intermediate step)
    - ResNet34 → ResNet18 + dendrites
    - BERT-base → BERT-small or DistilBERT + dendrites
    - GPT2-medium → GPT2-small + dendrites
    
    **Option 2: Switch to a more efficient architecture type**
    - ResNet50/34 → MobileNetV2 or EfficientNet-B0 + dendrites
    - BERT-base → DistilBERT + dendrites
    - ViT-Base → MobileViT or EfficientNet + dendrites
    
    Which would you like to try?"
  
  - **Wait for their choice before making any changes**
  
  - **After they choose**, make the appropriate changes:
    - **If model was configurable via arguments:** Modify their script's default argument or tell them which argument to change
    - **If model was hardcoded:** Modify their script to load the chosen model
  
  - **After making changes, tell them:**
    "I've updated your script to use [ChosenModel]. Before we add dendrites, please run a quick training test to confirm it loads correctly and trains without errors. This smaller model will have lower accuracy initially, which is expected. Dendrites will help recover performance."
  
  - Wait for confirmation before proceeding to Step 2
  
  **If they're already using the smallest variant within their model family (ResNet18, BERT-small, GPT2-small, etc.):**
  - Tell them: "You're using [CurrentModel], which is the smallest in its family. For maximum efficiency, I recommend switching to a fundamentally more efficient architecture type and adding dendrites.
    
    Important: Dendritic optimization often allows a smaller, more efficient architecture to achieve the accuracy goals you would have previously needed a larger model for. The dendrites add targeted capacity exactly where needed.
    
    Recommended switches:
    - ResNet18 → MobileNetV2 or EfficientNet-B0 + dendrites
    - BERT-small → DistilBERT + dendrites
    - ViT-Small → MobileViT or EfficientNet + dendrites
    
    Would you like to make this switch?"
  
  - **Wait for their decision before making any changes**
  
  - **If they agree**, make the appropriate changes:
    - **If model was configurable via arguments:** Modify their script's default argument or tell them which argument to change
    - **If model was hardcoded:** Modify their script to load the more efficient architecture
  
  - **After making changes, tell them:**
    "I've updated your script to use [MoreEfficientArchitecture]. Before we add dendrites, please run a quick training test to confirm it loads correctly and trains without errors. This architecture is designed for efficiency and will have lower accuracy initially. Dendrites will help recover performance."
  
  - Wait for confirmation before proceeding to Step 2
  
  **For Custom Models:**
  - **First, check if configuration is via arguments or hardcoded:**
    - Look for command-line arguments like `--num_layers`, `--hidden_dim`, `--width`, `--depth`, `--num_heads`, `--embed_dim`, etc.
    - Look for config file options for these settings
    
  - **If configurable via arguments/config:**
    - **ASK about their current configuration:** "What are your current model settings? For example, how many layers are you using? What are the hidden dimensions or channel counts?"
    - Wait for their answer
    
  - **If hardcoded:**
    - Analyze their model architecture to identify:
      - Number of layers (depth)
      - Hidden dimensions/channels (width)
    - Tell them: "I can see your model has [X] layers with hidden dimension [Y]."
  
  - **After understanding their current settings**, tell them: "To optimize for efficiency, let's make your model smaller first, then add dendrites.
    
    Important: Dendritic optimization often allows a smaller architecture to achieve the accuracy goals you would have previously needed a larger model for. The dendrites add targeted capacity exactly where needed, making smaller models surprisingly powerful.
    
    We can:
    - Reduce layer count (make it shallower)
    - Reduce hidden dimensions/channels (make it less wide)
    - Both"
  
  - Ask: "Would you like to reduce depth (fewer layers), width (fewer channels/neurons), or both? And what values would you like to use?"
  
  - **Wait for their decision on what to reduce and to what values**
  
  - **After they specify what they want**, implement the changes based on their code structure:
  
    **If they already have configurable width/depth parameters:**
    - Identify where these are set (config file, command-line args, hardcoded values)
    - Update them to the values they specified. For example:
      - `hidden_dim=512` → `hidden_dim=256`
      - `num_layers=12` → `num_layers=6`
    - Tell them what you changed
    
    **If they have command-line arguments but not for width/depth:**
    - Add new command-line arguments for the parameters they want to adjust
    - Example: Add `--hidden_dim`, `--num_layers`, `--num_channels`, etc.
    - Set defaults to the values they specified
    - Update their model instantiation to use these new arguments
    - Tell them what you changed
    
    **If they don't have command-line arguments:**
    - Add configurable settings at the top of their script or in a config section
    - Example:
      ```python
      # Model configuration (adjust these for efficiency)
      NUM_LAYERS = 6  # Original: 12
      HIDDEN_DIM = 256  # Original: 512
      NUM_CHANNELS = 32  # Original: 64
      ```
    - Update their model definition to use these variables instead of hardcoded values
    - Tell them what you changed
  
  - **Before proceeding to Step 2:**
    - Tell them: "I've updated your model to be smaller. Before we add dendrites, please run your training script now to confirm:
      1. Training runs without errors
      2. The model is indeed smaller (check parameter count)
      3. Accuracy is somewhat lower than your original model (expected)
      
      This establishes a baseline. After we add dendrites, we'll aim to recover or exceed your original accuracy with this more efficient architecture."
    
    - Wait for them to confirm they've tested the smaller model before proceeding to dendrite integration

