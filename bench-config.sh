#!/usr/bin/env bash
# Shared settings for run-benchmark.sh and run-memory-profile.sh. Both source
# this file, so a run is configured once here and the two stay in sync. Which
# of the two you call decides speed or memory; memory also forces N_REPEAT=1.

DESC_DIR="/CODES/DESC"
# BRANCHES=("master" "v0.17.3" "v0.17.2" "v0.17.1" "v0.17.0" "v0.16.0")
# BRANCHES=("v0.16.0" "v0.15.0" "v0.14.2" "v0.14.1" "v0.14.0" "v0.13.0" "v0.12.3" "v0.12.2")
BRANCHES=("v0.12.2")
# conda envs to compare (e.g. different dependency versions).
# Provide proper envs for your device here.
ENVS=("gpu")
DEVICE="gpu"           # cpu or gpu, must match the envs above
N_REPEAT=5             # will be overwritten to 1 for memory
INTERVAL=1e-5          # memory profiling: seconds between samples
RESULTS_DIR="results"  # one folder per branch below this

SCRIPT="01_eq_solve.py"
# SCRIPT="02_boundary_error.py"
# SCRIPT="03_prox_jac_freeb_coils.py"
# SCRIPT="04_quadratic_flux_jac.py"
# SCRIPT="05_prox_jac_freeb.py"
# SCRIPT="06_prox_jac_qa.py"
# SCRIPT="07_prox_jac_qa_coils.py"
# SCRIPT="08_fieldline_trace.py"
# SCRIPT="09_particle_trace.py"

# sourced, so these exit the driver that sourced this file
case "$DEVICE" in
    cpu|gpu) ;;
    *) echo "DEVICE must be 'cpu' or 'gpu', got '$DEVICE'"; exit 1 ;;
esac
if [ ! -f "scripts/$SCRIPT" ]; then
    echo "no such script: scripts/$SCRIPT"; exit 1
fi
