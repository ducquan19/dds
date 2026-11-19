# Sourced by every slurm/*.sh script. No conda: plain Python venv.
#   VENV     : venv location           (default: <repo>/.venv)
#   MODULES  : modules to load first   (e.g. MODULES="python/3.10 cuda/12.1"; empty = none)
if [ -n "${MODULES:-}" ] && command -v module >/dev/null 2>&1; then
  module load $MODULES
fi
VENV=${VENV:-$(pwd)/.venv}
activate_env() {
  [ -f "$VENV/bin/activate" ] || { echo "venv not found at $VENV - run: bash slurm/setup.sh"; exit 1; }
  source "$VENV/bin/activate"
}
