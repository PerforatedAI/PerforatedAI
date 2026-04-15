#!/bin/bash

# PerforatedAI Skill Installation Script
# Copies the PerforatedAI skill to your project, excluding large working directories

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if target path is provided
if [ -z "$1" ]; then
    echo -e "${RED}Error: Target project path required${NC}"
    echo "Usage: bash install-skill.sh /path/to/your/project"
    exit 1
fi

TARGET_PROJECT="$1"

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Validate target directory
if [ ! -d "$TARGET_PROJECT" ]; then
    echo -e "${RED}Error: Target directory does not exist: $TARGET_PROJECT${NC}"
    exit 1
fi

# Create .github/skills directory in target if it doesn't exist
mkdir -p "$TARGET_PROJECT/.github/skills"

echo -e "${GREEN}Installing PerforatedAI skills...${NC}"
echo "Source: $SCRIPT_DIR/.."
echo "Target: $TARGET_PROJECT/.github/skills"

# Copy skill files, dereferencing symlinks but excluding large working directories
rsync -rL \
    --exclude='ENV' \
    --exclude='lib' \
    --exclude='lib64' \
    --exclude='wandb' \
    --exclude='data' \
    --exclude='*.pth' \
    --exclude='*.pt' \
    --exclude='*.ckpt' \
    --exclude='old' \
    --exclude='PB*' \
    --exclude='PAI' \
    --exclude='out' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --info=progress2 \
    "$SCRIPT_DIR/../" "$TARGET_PROJECT/.github/skills/"

echo -e "${GREEN}✓ PerforatedAI skills installed successfully!${NC}"
echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo "1. Open VS Code in your project: code $TARGET_PROJECT"
echo "2. Open GitHub Copilot Chat"
echo "3. Say one of the following based on where you're at:"
echo "   - 'Perforate my model' - Start fresh integration"
echo "   - 'Debug my perforated model' - Fix issues with existing integration"
echo "   - 'Analyze my perforated results' - Review training outputs"
echo ""
echo "The PerforatedAI skill will guide you through the appropriate workflow."
