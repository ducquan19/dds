#!/bin/bash
#SBATCH --job-name=cascade_rs
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
# Generic GPU job. Normally submitted by slurm/submit_all.sh, but usable directly:
#   sbatch slurm/job.sh orig    0.25
#   sbatch slurm/job.sh probe   0.25          # probe + calibrate_tau -> out/tau_0.25.txt
#   sbatch slurm/job.sh dds     0.25
#   sbatch slurm/job.sh cheap   0.25
#   sbatch slurm/job.sh cascade 0.25 auto     # tau read from out/tau_0.25.txt
#   sbatch slurm/job.sh cascade 0.25 0.90     # fixed tau (Pareto sweep)
# Partition/account: export SBATCH_PARTITION=... SBATCH_ACCOUNT=... before sbatch.
set -euo pipefail
MODE=$1; SIGMA=$2; TAU=${3:-auto}
N=${N:-10000}; SKIP=${SKIP:-20}; BATCH=${BATCH:-500}; ALPHA=${ALPHA:-0.0005}
DELTA=${DELTA:-0.01}; CAL_ALPHA=${CAL_ALPHA:-0.05}; NCAL=${NCAL:-9500}

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
source slurm/env.sh
activate_env
export PYTHONPATH=cifar10 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
SIG=$(printf "%.2f" "$SIGMA")
CKPT=models/cifar10/resnet110/noise_${SIG}/checkpoint.pth.tar
mkdir -p out logs
echo "[$(date)] host=$(hostname) mode=$MODE sigma=$SIG tau=$TAU N=$N skip=$SKIP"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

case $MODE in
  orig)
    python cifar10/certify.py --sigma "$SIGMA" --skip "$SKIP" --N0 100 --N "$N" --alpha "$ALPHA" \
      --batch_size "$BATCH" --outfile "out/orig_${SIG}.tsv" ;;
  probe)
    python cifar10/probe.py --sigma "$SIGMA" --cheap_ckpt "$CKPT" --num_images "$NCAL" --K 4 \
      --split test_holdout --certify_skip "$SKIP" --batch_size "$BATCH" --out "out/probe_${SIG}.npz"
    python cifar10/calibrate_tau.py "out/probe_${SIG}.npz" --delta "$DELTA" --alpha "$CAL_ALPHA" \
      --out "out/tau_${SIG}.txt" ;;
  dds|cheap)
    python cifar10/certify_cascade.py --mode "$MODE" --sigma "$SIGMA" --cheap_ckpt "$CKPT" \
      --skip "$SKIP" --N "$N" --alpha "$ALPHA" --batch_size "$BATCH" --outfile "out/${MODE}_${SIG}.tsv" ;;
  cascade)
    if [ "$TAU" = auto ]; then TAU=$(cat "out/tau_${SIG}.txt"); TAG=cal; else TAG=$TAU; fi
    echo "using tau=$TAU"
    python cifar10/certify_cascade.py --mode cascade --tau "$TAU" --sigma "$SIGMA" --cheap_ckpt "$CKPT" \
      --skip "$SKIP" --N "$N" --alpha "$ALPHA" --batch_size "$BATCH" \
      --outfile "out/cascade_${SIG}_tau${TAG}.tsv" ;;
  *) echo "unknown mode $MODE"; exit 1 ;;
esac
echo "[$(date)] done"
