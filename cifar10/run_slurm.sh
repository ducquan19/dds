#!/bin/bash
#SBATCH --job-name=cascade_rs
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
# Usage: sbatch run_slurm.sh <mode: dds|cheap|cascade|orig|probe> <sigma> [tau]
set -e
MODE=$1; SIGMA=$2; TAU=${3:-0.95}
source ~/.bashrc && conda activate dds
cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH=cifar10
SIG=$(printf "%.2f" "$SIGMA")
CKPT=models/cifar10/resnet110/noise_${SIG}/checkpoint.pth.tar
mkdir -p out logs

case $MODE in
  orig)  python cifar10/certify.py --sigma $SIGMA --skip 20 --N0 100 --N 10000 --alpha 0.0005 \
           --batch_size 500 --outfile out/orig_${SIG}.tsv ;;
  probe) python cifar10/probe.py --sigma $SIGMA --cheap_ckpt $CKPT --num_images 5000 --K 4 \
           --out out/probe_${SIG}.npz ;;
  *)     python cifar10/certify_cascade.py --mode $MODE --tau $TAU --sigma $SIGMA --cheap_ckpt $CKPT \
           --skip 20 --N 10000 --alpha 0.0005 --outfile out/${MODE}_${SIG}.tsv ;;
esac
