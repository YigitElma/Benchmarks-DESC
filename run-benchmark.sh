#!/usr/bin/env bash
set -euo pipefail

# run from this script's directory, so scripts/ and results/ always resolve
cd "$(dirname "$0")"

# branches, envs, device, script, ... are all set here. run-all.sh points
# BENCH_CONFIG at its own file instead.
source "${BENCH_CONFIG:-./bench-config.sh}"

PROFILE_MODE="speed"

source "$(conda info --base)/etc/profile.d/conda.sh"

echo ""
echo "Running scripts/$SCRIPT on $DEVICE ($PROFILE_MODE mode, N_REPEAT=$N_REPEAT)"
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

        echo "=== Benchmarking branch: $BRANCH (env: $ENV) ==="
        git -C "$DESC_DIR" checkout "$BRANCH"
        python "scripts/$SCRIPT" "$DEVICE" "$PROFILE_MODE" "$N_REPEAT" \
            "$RESULTS_DIR/$LABEL"
        echo ""
    done
done

echo "=== Comparison ==="
python compare-results.py "${LABELS[@]}" --mode "$PROFILE_MODE" --device "$DEVICE" \
    --script "${SCRIPT%.py}" --results-dir "$RESULTS_DIR"
