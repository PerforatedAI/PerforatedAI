# PerforatedAI Agent Skills

These are [Agent Skills](https://github.com/anthropics/skills) - reusable, task-triggered instructions in the portable `SKILL.md` format. The **same files work across Claude Code, GitHub Copilot, OpenAI Codex, and Cursor**; only the install location differs (handled for you below).

They teach your AI coding agent how to integrate [PerforatedAI](https://github.com/PerforatedAI/PerforatedAI) - adding artificial dendrites to a PyTorch model - directly in _your own_ project.

## What's here

| Skill                                                                                   | Say / when it triggers                                                                                         |
| --------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| **[perforatedai](perforatedai/SKILL.md)**                                               | `"Perforate my model"` to start setup, or `"Debug my perforated model"`. The main entry point - start here.    |
| **[perforatedai-analyze](perforatedai-analyze/SKILL.md)**                               | `"Analyze my perforated results"` after training completes. Reviews CSV outputs and recommends config changes. |
| **[perforatedai-distributed](perforatedai-distributed/SKILL.md)**                       | Auto-loaded by the main skill when DataParallel / DDP multi-GPU training is detected.                          |
| **[perforatedai-libraries-transformers](perforatedai-libraries-transformers/SKILL.md)** | Auto-loaded when your script uses the HuggingFace `Trainer`.                                                   |
| **[perforatedai-wandb](perforatedai-wandb/SKILL.md)**                                   | When you want Weights & Biases sweeps/logging with PerforatedAI.                                               |
| **[perforatedai-complex-methods](perforatedai-complex-methods/SKILL.md)**               | Edge cases such as AMP / `GradScaler` crashes in p mode.                                                       |

You normally only invoke **perforatedai**; it pulls in the others as needed.

## Prerequisite

The skills edit _your_ training script, but the library itself must be installed in your Python environment:

```bash
pip install perforatedai perforatedbp
```

## Install the skills into your project

### Option A - one-liner (recommended)

The [`skills` CLI](https://github.com/vercel-labs/skills) copies them into the right folder for your agent automatically. Run this from your project root and pick your agent with `-a`. The `-s '*'` installs all skills without prompting (drop it to choose from an interactive picker instead):

```bash
# Claude Code
npx skills add PerforatedAI/PerforatedAI -a claude-code -s '*'

# GitHub Copilot
npx skills add PerforatedAI/PerforatedAI -a github-copilot -s '*'

# OpenAI Codex
npx skills add PerforatedAI/PerforatedAI -a codex -s '*'

# Cursor
npx skills add PerforatedAI/PerforatedAI -a cursor -s '*'
```

Add multiple agents in one call (`-a claude-code -a cursor`). To install just one skill, point at its subdirectory:

```bash
npx skills add https://github.com/PerforatedAI/PerforatedAI/tree/main/skills/perforatedai -a claude-code
```

### Option B - manual copy

No Node? Copy the `skills/` subdirectories into the folder your agent reads:

| Agent          | Destination in your project              |
| -------------- | ---------------------------------------- |
| Claude Code    | `.claude/skills/`                        |
| GitHub Copilot | `.github/skills/` (or `.claude/skills/`) |
| OpenAI Codex   | `.codex/skills/`                         |
| Cursor         | `.cursor/skills/` (or `.claude/skills/`) |

For example, for Claude Code:

```bash
mkdir -p .claude/skills
# from a clone of this repo:
cp -R /path/to/PerforatedAI/skills/* .claude/skills/
```

`.claude/skills/` is read by Claude Code and honored as a fallback by Copilot and Cursor, so it's the best single target if you use more than one of those.

## Use them

Open your agent in your project and say one of the trigger phrases above - e.g. **"Perforate my model"**. The skill walks you through analyzing your model, adding the ~10 lines of PerforatedAI integration, and verifying dendrites are training.
