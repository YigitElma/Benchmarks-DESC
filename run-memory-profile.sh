#!/usr/bin/env bash
set -euo pipefail

# run from this script's directory, so scripts/ and results/ always resolve
cd "$(dirname "$0")"

# branches, envs, device, script, ... are all set here. run-all.sh points
# BENCH_CONFIG at its own file instead.
source "${BENCH_CONFIG:-./bench-config.sh}"

# this driver always profiles memory, with a single repeat
PROFILE_MODE="memory"
N_REPEAT=1

source "$(conda info --base)/etc/profile.d/conda.sh"

echo ""
echo "Profiling scripts/$SCRIPT memory on $DEVICE (N_REPEAT=$N_REPEAT)"
echo ""

LABELS=()
for ENV in "${ENVS[@]}"; do
    conda activate "$ENV"
    echo "########## Environment: $ENV ##########"
    for BRANCH in "${BRANCHES[@]}"; do
        # results go to one folder per branch, / -> -. With several envs the env
        # is prefixed, so the two runs of a branch do not overwrite each other.
        LABEL="${BRANCH//\//-}"
        if [ "${#ENVS[@]}" -gt 1 ]; then
            LABEL="${ENV}-${LABEL}"
        fi
        LABELS+=("$LABEL")

        echo "=== Profiling branch: $BRANCH (env: $ENV) ==="
        git -C "$DESC_DIR" checkout "$BRANCH"
        python memory-profile.py "$DEVICE" "$RESULTS_DIR/$LABEL" \
            "scripts/$SCRIPT" "$N_REPEAT" "$INTERVAL"
        echo ""
    done
done

echo "=== Comparison ==="
python compare-results.py "${LABELS[@]}" --mode memory --device "$DEVICE" \
    --script "${SCRIPT%.py}" --results-dir "$RESULTS_DIR"
