#!/bin/sh
# PerforatedAI Studio installer.
#
# Run from inside the project you want to install into:
#   curl -fsSL https://raw.githubusercontent.com/PerforatedAI/PerforatedAI/main/Studio_Install/bootstrap.sh | sh
#
# Re-run it to upgrade. It is idempotent.
#
# This script runs on the HOST, not in the container: it calls docker and writes
# .mcp.json, neither of which a container can do for us.
set -e

# GHCR, not Docker Hub: Docker Hub rate-limits anonymous pulls per IP, and the
# first thing this script does is an anonymous pull. A team behind one corporate
# NAT would hit `toomanyrequests` on a machine where nothing is wrong (ADR 0022).
IMAGE_REPO="ghcr.io/perforatedai/studio"
LABEL_KEY="org.opencontainers.image.version"

VERSION="latest"
PORT=3002
IMAGE=""

while [ $# -gt 0 ]; do
  case "$1" in
    --version) VERSION="$2"; shift 2 ;;
    --port)    PORT="$2"; shift 2 ;;
    # Dev hatch: install from a local image instead of pulling. Undocumented.
    --image)   IMAGE="$2"; shift 2 ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

# --image names an image that already exists locally (a dev build): use it as-is
# and skip the pull. Otherwise we install a published version from the registry.
if [ -n "$IMAGE" ]; then
  LOCAL_IMAGE=1
else
  LOCAL_IMAGE=0
  IMAGE="$IMAGE_REPO:$VERSION"
fi

# --- pull. Surface network/registry failures here, at install time, rather than
# in the invisible MCP subprocess later.
if [ "$LOCAL_IMAGE" -eq 0 ]; then
  if ! docker pull "$IMAGE"; then
    printf 'error: failed to pull %s — check your network and that Docker is running\n' "$IMAGE" >&2
    exit 1
  fi
fi

# --- resolve. `latest` is an INPUT to this installer, never an output: the image
# reference we write into .mcp.json is re-resolved by Claude Code on every session
# start, so a floating tag there would let the MCP Server drift to a new version
# while the skills we copy to disk below stay frozen at this one (ADR 0023).
#
# The version rides inside the image as a label, so reading it back needs no
# registry API and no second network call.
RESOLVED="$(docker inspect --format "{{index .Config.Labels \"$LABEL_KEY\"}}" "$IMAGE")"
if [ -z "$RESOLVED" ]; then
  printf 'error: %s carries no %s label — cannot determine its version\n' "$IMAGE" "$LABEL_KEY" >&2
  exit 1
fi
# What goes into .mcp.json. For a local dev build that's the image itself — wiring
# up a published tag the local build isn't would point every session at the wrong
# image (or none at all).
if [ "$LOCAL_IMAGE" -eq 1 ]; then
  PINNED="$IMAGE"
else
  PINNED="$IMAGE_REPO:$RESOLVED"
  # Give the daemon the image under the name everything else refers to. Pulling
  # `:latest` stores it under THAT tag, but .mcp.json and the manifest name the
  # resolved tag — so without this the daemon holds no image called
  # `…:v0.1.0`, uninstall's `docker rmi` finds nothing and the image leaks, and
  # the user's first session re-pulls what is already on disk.
  #
  # Then drop the tag we pulled under, so the resolved ref is the ONLY local
  # reference. `docker rmi <tag>` deletes the tag, not the image, while any other
  # tag still points at it — leave `:latest` behind and uninstall's rmi succeeds
  # while the image quietly stays on disk.
  if [ "$IMAGE" != "$PINNED" ]; then
    docker tag "$IMAGE" "$PINNED"
    docker rmi "$IMAGE" >/dev/null 2>&1 || true
    # Everything below must now refer to the pinned ref. `docker run` and
    # `docker create` silently PULL an image that isn't present locally, so a
    # later command still naming `:latest` would fetch it again and recreate the
    # very tag we just dropped.
    IMAGE="$PINNED"
  fi
fi

# --- smoke test: verify the image actually runs AND its deps import on this
# machine before we wire up the MCP config. Importing the module exercises the
# heavy imports (torch, the ml plugin) without starting the blocking server.
# A failure here is loud; the same failure inside the MCP subprocess later is not.
if ! docker run --rm --entrypoint python "$IMAGE" -c "import mcp_server.server" >/dev/null 2>&1; then
  echo "error: $IMAGE failed to start on this machine. Details:" >&2
  docker run --rm --entrypoint python "$IMAGE" -c "import mcp_server.server" 2>&1 | sed 's/^/  /' >&2
  exit 1
fi

PWD_ABS="$(pwd)"
MCP_FILE=".mcp.json"
TOOLS_DIR="$PWD_ABS/.perforated_tools"
SKILLS_DIR="$PWD_ABS/.claude/skills"

MANIFEST="$TOOLS_DIR/installed.json"

# --- reconcile. Re-running this script is the upgrade path, so a version that
# renames or drops a skill must take the old one away with it — otherwise it sits
# orphaned in .claude/skills/, still loaded by Claude, calling tools that may no
# longer exist.
#
# We remove ONLY the skills our own manifest says we installed. Never the whole of
# .claude/skills/ — the user keeps their own skills in there, and eating a
# customer's hand-written work on upgrade is unforgivable. The manifest is the
# only thing that lets us tell our skills from theirs.
if [ -f "$MANIFEST" ]; then
  PREVIOUS_SKILLS="$(python3 -c 'import json,sys; print(" ".join(json.load(open(sys.argv[1])).get("skills", [])))' "$MANIFEST")"
  for skill in $PREVIOUS_SKILLS; do
    rm -rf "$SKILLS_DIR/$skill"
  done
fi

# --- extract the Package the image carries: the skills, the runtime launcher and
# the uninstaller (ADR 0021). /app/package/ is a frozen contract — this script is
# always published from main but may be run against an image cut long ago.
#
# `docker cp` rather than a `-v` bind mount: a container writing through a bind
# mount leaves root-owned files on the host on Linux. cp chowns to the caller.
#
# Skills land in a staging dir first, so we learn exactly which skills came out of
# the image. Listing .claude/skills/ afterwards would sweep the user's own skills
# into our manifest — and a later uninstall would then delete them.
mkdir -p "$TOOLS_DIR" "$SKILLS_DIR"
STAGE="$(mktemp -d)"
CID="$(docker create "$IMAGE")"
docker cp "$CID:/app/package/skills/." "$STAGE"
docker cp "$CID:/app/package/dashboard-run.sh" "$TOOLS_DIR/dashboard-run.sh"
docker cp "$CID:/app/package/uninstall.sh" "$TOOLS_DIR/uninstall.sh"
docker rm "$CID" >/dev/null
chmod +x "$TOOLS_DIR/dashboard-run.sh" "$TOOLS_DIR/uninstall.sh"

OUR_SKILLS=""
for skill_path in "$STAGE"/*/; do
  [ -d "$skill_path" ] || continue
  skill_name="$(basename "$skill_path")"
  rm -rf "$SKILLS_DIR/$skill_name"
  cp -R "$skill_path" "$SKILLS_DIR/$skill_name"
  OUR_SKILLS="$OUR_SKILLS $skill_name"
done
rm -rf "$STAGE"

[ -f "$MCP_FILE" ] || echo '{}' > "$MCP_FILE"

# Write or overwrite the dashboard mcpServers entry, preserving other keys.
python3 - "$MCP_FILE" "$PORT" "$PWD_ABS" "$PINNED" <<'EOF'
import json
import sys

file, port, cwd, image = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
with open(file) as f:
    config = json.load(f)
config.setdefault("mcpServers", {})
config["mcpServers"]["dashboard"] = {
    "command": f"{cwd}/.perforated_tools/dashboard-run.sh",
    "args": [
        "run", "--rm", "-i",
        "-p", f"{port}:{port}",
        "-v", f"{cwd}:/workspace:ro",
        "-v", f"{cwd}/.perforated_tools:/perforated_tools:rw",
        image,
    ],
}
with open(file, "w") as f:
    json.dump(config, f, indent=2)
    f.write("\n")
EOF

# --- record what we installed. This manifest is the record of what this installer
# OWNS: uninstall reads it to know what to tear down, and the next run reads it to
# know what to clean up before laying down a new version. Without it we would have
# to guess, and guessing means either leaving orphans behind or deleting skills the
# user wrote themselves.
python3 - "$MANIFEST" "$PINNED" "$RESOLVED" "$PORT" "$OUR_SKILLS" <<'EOF'
import datetime
import json
import sys

path, image, version, port, skill_names = sys.argv[1:6]
manifest = {
    "image": image,
    "version": version,
    "port": int(port),
    "skills": skill_names.split(),
    "installed_at": datetime.datetime.now(datetime.timezone.utc)
        .isoformat(timespec="seconds").replace("+00:00", "Z"),
}
with open(path, "w") as f:
    json.dump(manifest, f, indent=2)
    f.write("\n")
EOF

BOOTSTRAP_URL="https://raw.githubusercontent.com/PerforatedAI/PerforatedAI/main/Studio_Install/bootstrap.sh"

echo "Installed PerforatedAI Studio $RESOLVED on port $PORT"
echo "  skills:    $SKILLS_DIR"
echo "  uninstall: $TOOLS_DIR/uninstall.sh"
echo
# Re-running this script IS the upgrade path — there is no update.sh, because
# anything shipped in the image is by definition the previous version's logic.
# Nothing else tells the user this, so say it here.
echo "To upgrade, re-run the installer:"
echo "  curl -fsSL $BOOTSTRAP_URL | sh"
