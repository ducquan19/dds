#!/bin/bash
# Submit the full CIFAR-10 pipeline for one or more sigmas, from the repo root:
#   bash slurm/submit_all.sh 0.25            # first: sigma=0.25 only
#   bash slurm/submit_all.sh 0.25 0.50 1.00
# Per sigma:
#   orig   : original certify.py (reproduction check vs Dual RS numbers)
#   dds    : patched always-expensive baseline (the fair comparison for cascade)
#   cheap  : Cohen ResNet-110 alone
#   probe  : probe + calibrate_tau  -> out/tau_<sigma>.txt
#   cascade(cal)  : afterok:probe, uses the calibrated tau
#   cascade sweep : fixed taus for the Pareto plot (independent of probe)
# Env overrides: SWEEP="0.5 0.7 0.9 0.99"  N=10000  SKIP=20  BATCH=500  DELTA=0.01
#                SBATCH_PARTITION / SBATCH_ACCOUNT (read natively by sbatch)
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs out
SWEEP=${SWEEP:-"0.5 0.7 0.8 0.9 0.95 0.99"}
[ $# -ge 1 ] || { echo "usage: $0 SIGMA [SIGMA...]"; exit 1; }

sub() { sbatch --parsable --export=ALL "$@"; }

for S in "$@"; do
  SIG=$(printf "%.2f" "$S")
  j_orig=$(sub -J orig_$SIG    slurm/job.sh orig  "$S")
  j_dds=$(sub  -J dds_$SIG     slurm/job.sh dds   "$S")
  j_chp=$(sub  -J cheap_$SIG   --time=02:00:00 slurm/job.sh cheap "$S")
  j_prb=$(sub  -J probe_$SIG   --time=04:00:00 slurm/job.sh probe "$S")
  j_cal=$(sub  -J casc_${SIG}_cal --dependency=afterok:$j_prb slurm/job.sh cascade "$S" auto)
  echo "sigma=$SIG orig=$j_orig dds=$j_dds cheap=$j_chp probe=$j_prb cascade_cal=$j_cal"
  for T in $SWEEP; do
    j=$(sub -J casc_${SIG}_$T slurm/job.sh cascade "$S" "$T")
    echo "   sweep tau=$T -> $j"
  done
done
echo "monitor: squeue -u \$USER ; logs in logs/ ; results in out/"
echo "when done: bash slurm/report.sh $*"
