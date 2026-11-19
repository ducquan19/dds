#!/bin/bash
# Summarise results (runs on login node, CPU only):  bash slurm/report.sh 0.25 [0.50 1.00]
cd "$(dirname "$0")/.."
eval "$(conda shell.bash hook)"; conda activate "${CONDA_ENV:-dds}"
REF_DIR=${REF_DIR:-../Dual-Randomized-Smoothing/reproduce/data/carlini}
for S in "$@"; do
  SIG=$(printf "%.2f" "$S")
  echo "================ sigma=$SIG"
  [ -f "out/tau_${SIG}.txt" ] && echo "calibrated tau = $(cat out/tau_${SIG}.txt)"
  ARGS=()
  for REF in "$REF_DIR/${SIG}.tsv" "$REF_DIR/${S}.tsv"; do
    [ -f "$REF" ] && { ARGS+=("ref=$REF"); break; }
  done
  for F in out/orig_${SIG}.tsv out/dds_${SIG}.tsv out/cheap_${SIG}.tsv out/cascade_${SIG}_tau*.tsv; do
    [ -s "$F" ] || continue
    NAME=$(basename "$F" .tsv); NAME=${NAME%_${SIG}}; NAME=${NAME/cascade_${SIG}_/casc_}
    ARGS+=("$NAME=$F")
  done
  python cifar10/compare.py "${ARGS[@]}" --skip "${SKIP:-20}"
done
