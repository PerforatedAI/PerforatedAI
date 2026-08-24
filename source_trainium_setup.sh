#!/usr/bin/env bash
#
# trainium_setup_clean.sh — Trainium setup script
# USAGE: source source_trainium_setup.sh
# (Must be sourced to keep venv activation in your current shell)

# ----------------------------- CONFIG ---------------------------------------
ECR_ACCOUNT="421672808698"
ECR_REGION="us-east-1"
ECR_REPO="concourse-release-0461d3b"
ECR_TAG="latest"
IMAGE_REF="${ECR_ACCOUNT}.dkr.ecr.${ECR_REGION}.amazonaws.com/${ECR_REPO}:${ECR_TAG}"

# ----------------------------- AWS CREDENTIALS ------------------------------
echo "Checking AWS credentials..."
if ! aws sts get-caller-identity >/dev/null 2>&1; then
  echo "No AWS credentials found. Let's set them up now."
  echo "(You'll need your Access Key ID and Secret Access Key. Region should be: ${ECR_REGION})"
  aws configure
  # Re-check to make sure the entered credentials work
  aws sts get-caller-identity >/dev/null 2>&1
fi
# Make sure a region is set even if creds came from an instance role
if [ -z "$(aws configure get region 2>/dev/null || true)" ]; then
  echo "No default region set — setting to ${ECR_REGION}"
  aws configure set region "$ECR_REGION"
fi
echo "AWS credentials OK: account $(aws sts get-caller-identity --query Account --output text)"

# ----------------------------- PULL DOCKER IMAGE ----------------------------
echo "Logging into ECR..."
aws ecr get-login-password --region "$ECR_REGION" \
  | docker login --username AWS --password-stdin \
    "${ECR_ACCOUNT}.dkr.ecr.${ECR_REGION}.amazonaws.com"

echo "Pulling docker image: $IMAGE_REF"
docker pull "$IMAGE_REF"

echo "Done! Image pulled successfully."

# 3. Extract the workspace artifacts (loose wheels + runtime debs) out of the image
cd $HOME
imageID=$(docker images -q --filter reference=421672808698.dkr.ecr.us-east-1.amazonaws.com/concourse-release-0461d3b:latest)
docker create --name tmp $imageID && docker cp tmp:/workspace . && docker rm tmp

# 4. Install the runtime .debs on the host (DKMS kernel build ~10 min)
sudo dpkg -i $HOME/workspace/runtime_artifacts/*.deb

# 5. Build the venv from the extracted wheels
cd $HOME/workspace
python3.12 -m venv native_venv
source native_venv/bin/activate
pip install --upgrade pip          # stock pip 24.0 chokes on the nki version string, upgrade first
pip install neuron_torch_mlir_wheels/neuron_torch_mlir-*.whl
pip install nki_wheels/nki-*.whl               # actual wheel is nki-0.4.0b4*, use the glob
pip install neuronx_cc_wheels/neuronx_cc-*.whl
cd torch_neuron_eager && pip install -e . && cd ..
pip uninstall -y torch
pip install torch==2.12.1 --index-url https://download.pytorch.org/whl/cpu
python -c "import torch, torch_neuronx; print(torch.randn(4,4, device='neuron'))"
pip install transformers

export NEURON_LAUNCH_BLOCKING=1
export TORCH_NEURONX_LOG_LEVEL=3
export TORCH_NEURONX_ENABLE_STACK_TRACE=1
export NEURON_CC_FLAGS="--verbose=INFO"
