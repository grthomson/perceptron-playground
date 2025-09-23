# scripts/run_entity_resolution_demo.sh
#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-python}"

# Choose features (use 2 for 2D plot, 3 for 3D plot)
COLS="${COLS:-forename,surname,postcode}"
NEG_RATIO="${NEG_RATIO:-1.0}"
ETA="${ETA:-0.1}"
N_ITER="${N_ITER:-10}"
SEED="${SEED:-7}"

echo "==> Generating clean + noisy toy data"
"$PY" "$ROOT/scripts/make_toy_clean_dataset.py"
"$PY" "$ROOT/scripts/make_noisy_copy.py"

echo "==> Training perceptron on cols: $COLS"
"$PY" "$ROOT/scripts/train_linkage_perceptron.py" \
  --cols "$COLS" \
  --neg-ratio "$NEG_RATIO" \
  --eta "$ETA" \
  --n-iter "$N_ITER" \
  --seed "$SEED"

echo "==> Plotting (learning curve + 2D/3D boundary if applicable)"
"$PY" "$ROOT/scripts/plot_entity_resolution.py"
