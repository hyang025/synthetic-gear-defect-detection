#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
REPO_DIR="$(cd -- "$PROJECT_DIR/../.." && pwd)"
BUILD_CACHE="$REPO_DIR/.cache/lisa-two-belt-r06"
mkdir -p "$BUILD_CACHE"
unset LISA_ASSEMBLY_ONLY LISA_REBUILD_PARTS LISA_VALIDATE_PARTS
rm -f -- "$BUILD_CACHE/freecad_complete"
export CAD_SMOKE_SCRIPT="$SCRIPT_DIR/generate_two_belt.py"
export CAD_SMOKE_OUTPUT="$BUILD_CACHE"
"$REPO_DIR/scripts/xvfb-run.sh" "$REPO_DIR/scripts/freecad.sh" \
  "$REPO_DIR/scripts/smoke_freecad.FCMacro" > "$BUILD_CACHE/freecad.log" 2>&1
if [[ ! -f "$BUILD_CACHE/freecad_complete" ]]; then
  cat "$BUILD_CACHE/freecad_python.log"
  exit 1
fi
"$REPO_DIR/.venv/bin/python" "$SCRIPT_DIR/validate_two_belt.py"
export LISA_RENDER_TAG=r06
export LISA_ASSEMBLY_REPORT=two_belt_r06_assembly_report.json
"$REPO_DIR/scripts/blender.sh" --background --factory-startup --python-exit-code 1 \
  --python "$SCRIPT_DIR/render_two_belt_draft.py" > "$BUILD_CACHE/blender.log" 2>&1
"$REPO_DIR/.venv/bin/python" "$SCRIPT_DIR/label_two_belt_draft.py"
"$REPO_DIR/.venv/bin/python" "$SCRIPT_DIR/generate_two_belt_manual.py"
"$REPO_DIR/.venv/bin/python" "$SCRIPT_DIR/document_two_belt.py"
"$REPO_DIR/.venv/bin/python" "$SCRIPT_DIR/package_two_belt.py"
