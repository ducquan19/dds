# Sourced by every slurm/*.sh script. No conda: plain Python venv.
#   VENV     : venv location           (default: <repo>/.venv)
#   MODULES  : modules to load first   (e.g. MODULES="python/3.10 cuda/12.1"; empty = none)
if [ -n "${MODULES:-}" ] && command -v module >/dev/null 2>&1; then
  module load $MODULES
fi
VENV=${VENV:-$(pwd)/.venv}
# $HOME/.cache may be read-only on this cluster -> keep every cache inside the repo
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-$(pwd)/.cache}
export HF_HOME=${HF_HOME:-$XDG_CACHE_HOME/huggingface}
export TORCH_HOME=${TORCH_HOME:-$XDG_CACHE_HOME/torch}
export PIP_CACHE_DIR=${PIP_CACHE_DIR:-$XDG_CACHE_HOME/pip}
mkdir -p "$HF_HOME" "$TORCH_HOME" "$PIP_CACHE_DIR"
activate_env() {
  [ -f "$VENV/bin/activate" ] || { echo "venv not found at $VENV - run: bash slurm/setup.sh"; exit 1; }
  source "$VENV/bin/activate"
}
