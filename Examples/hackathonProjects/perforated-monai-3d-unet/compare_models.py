import torch
import pandas as pd
from monai.networks.nets import UNet

# ========================
# CONFIG
# ========================
DEVICE = "cpu"   # no GPU needed

# Paths to trained models
MODELS = {
    "Baseline (Full)": {
        "path": "checkpoints/baseline/unet_baseline_new.pt",
        "channels": (32, 64, 128, 256),
    },
    "Dendritic (Full)": {
        "path": "checkpoints/dendritic/unet_dendritic_new.pt",
        "channels": (32, 64, 128, 256),
    },
    "Baseline (Compressed)": {
        "path": "checkpoints/baseline/unet_baseline_compressed_new.pt",
        "channels": (24, 40, 80, 160),
    },
    "Dendritic (Compressed)": {
        "path": "checkpoints/dendritic/unet_dendritic_compressed_new.pt",
        "channels": (24, 40, 80, 160),
    },
}

NUM_CLASSES = 4
IN_CHANNELS = 4


# ========================
# HELPERS
# ========================
def build_unet(channels):
    return UNet(
        spatial_dims=3,
        in_channels=IN_CHANNELS,
        out_channels=NUM_CLASSES,
        channels=channels,
        strides=(2, 2, 2),
        num_res_units=2,
    )


def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ========================
# MAIN
# ========================
results = []

for name, cfg in MODELS.items():
    print(f"\n🔍 Loading: {name}")

    model = build_unet(cfg["channels"]).to(DEVICE)
    state = torch.load(cfg["path"], map_location=DEVICE)
    model.load_state_dict(state, strict=False)

    params = count_trainable_params(model)
    params_m = params / 1e6

    results.append({
        "Model": name,
        "Channels": str(cfg["channels"]),
        "Parameters": params,
        "Parameters (Millions)": round(params_m, 3),
    })

    print(f"  Parameters: {params:,} ({params_m:.3f}M)")


# ========================
# SAVE RESULTS
# ========================
df = pd.DataFrame(results)

print("\n📊 Model Comparison")
print(df)

df.to_csv("model_comparison.csv", index=False)
print("\n✅ Saved: model_comparison.csv")
