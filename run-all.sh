#!/usr/bin/env bash
# Full sweep, e.g. to run weekly: every script in scripts/, on CPU and GPU, in
# speed and memory mode, for each commit or branch given.
#
#   ./run-all.sh                        # the COMMITS below
#   ./run-all.sh master yge/prox-fxh    # these instead
#
# Standalone: it writes its own config for the drivers and never reads
# bench-config.sh, so whatever is set up there for interactive runs is left
# alone. A run that fails or hangs is logged and skipped, the sweep continues,
# and the failures are listed at the end together with the comparison tables.

# no -e: one failing run must not abort the sweep
set -uo pipefail

# run from this script's directory, so scripts/ and results/ always resolve
cd "$(dirname "$0")"

# --- Configuration: edit these as needed ---
DESC_DIR="/CODES/DESC"
COMMITS=("master")                           # used when no argument is given
DEVICES=("cpu" "gpu")
MODES=("speed" "memory")
declare -A ENV_OF=([cpu]="cpu" [gpu]="gpu")  # conda env of each device
N_REPEAT=5                                   # memory runs always use 1
INTERVAL=1e-5                                # memory: seconds between samples
TIMEOUT=3600                                 # per run, seconds
RESULTS_DIR="results-all"
# --------------------------------------------

if [ "$#" -gt 0 ]; then
    COMMITS=("$@")
fi

SCRIPTS=()
for f in scripts/*.py; do
    case "$(basename "$f")" in
        __init__.py|universal.py) continue ;;  # helpers, not benchmarks
    esac
    SCRIPTS+=("$(basename "$f")")
done
if [ "${#SCRIPTS[@]}" -eq 0 ]; then
    echo "no benchmark scripts in scripts/"; exit 1
fi

LABELS=()
for COMMIT in "${COMMITS[@]}"; do
    LABELS+=("${COMMIT//\//-}")
done

LOG_DIR="$RESULTS_DIR/logs"
mkdir -p "$LOG_DIR"
STAMP="$(date +%F)"
SUMMARY="$RESULTS_DIR/summary-$STAMP.md"

# the drivers read this instead of bench-config.sh, rewritten for every run
TMP_CFG="$(mktemp)"
trap 'rm -f "$TMP_CFG"' EXIT

TOTAL=$((${#COMMITS[@]} * ${#DEVICES[@]} * ${#MODES[@]} * ${#SCRIPTS[@]}))
COUNT=0
FAILED=()
SWEEP_START=$SECONDS

echo ""
echo "Sweeping ${#SCRIPTS[@]} scripts x ${#DEVICES[@]} devices x ${#MODES[@]} modes"
echo "over ${#COMMITS[@]} commits: ${COMMITS[*]}"
echo "results in $RESULTS_DIR, logs in $LOG_DIR"
echo ""

for COMMIT in "${COMMITS[@]}"; do
    for DEVICE in "${DEVICES[@]}"; do
        for MODE in "${MODES[@]}"; do
            for SCRIPT in "${SCRIPTS[@]}"; do
                COUNT=$((COUNT + 1))
                TAG="${COMMIT//\//-} ${SCRIPT%.py} $DEVICE $MODE"
                LOG="$LOG_DIR/${SCRIPT%.py}_${DEVICE}_${MODE}_${COMMIT//\//-}.log"

                cat > "$TMP_CFG" <<EOF
DESC_DIR="$DESC_DIR"
BRANCHES=("$COMMIT")
ENVS=("${ENV_OF[$DEVICE]}")
DEVICE="$DEVICE"
N_REPEAT=$N_REPEAT
INTERVAL=$INTERVAL
RESULTS_DIR="$RESULTS_DIR"
SCRIPT="$SCRIPT"
EOF
                DRIVER="./run-benchmark.sh"
                if [ "$MODE" = "memory" ]; then
                    DRIVER="./run-memory-profile.sh"
                fi

                printf "[%3d/%3d] %-52s " "$COUNT" "$TOTAL" "$TAG"
                START=$SECONDS
                if BENCH_CONFIG="$TMP_CFG" timeout "$TIMEOUT" "$DRIVER" \
                        > "$LOG" 2>&1; then
                    echo "ok ($((SECONDS - START))s)"
                else
                    echo "FAILED ($((SECONDS - START))s) -> $LOG"
                    FAILED+=("$TAG")
                fi
            done
        done
    done
done

# the drivers activated their env in their own shell, get one for the tables
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_OF[${DEVICES[0]}]}"

{
    echo "# DESC benchmark sweep, $STAMP"
    echo ""
    echo "- commits: ${COMMITS[*]}"
    echo "- scripts: ${#SCRIPTS[@]}, devices: ${DEVICES[*]}, modes: ${MODES[*]}"
    echo "- N_REPEAT: $N_REPEAT, took $(((SECONDS - SWEEP_START) / 60)) min"
    echo "- failed: ${#FAILED[@]} of $TOTAL"
    for TAG in "${FAILED[@]+"${FAILED[@]}"}"; do
        echo "  - $TAG"
    done
    for DEVICE in "${DEVICES[@]}"; do
        for MODE in "${MODES[@]}"; do
            echo ""
            echo "## $MODE, $DEVICE"
            python compare-results.py "${LABELS[@]}" --mode "$MODE" \
                --device "$DEVICE" --results-dir "$RESULTS_DIR" --markdown
        done
    done
} | tee "$SUMMARY"

echo ""
echo "summary written to $SUMMARY"
