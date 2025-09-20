#!/usr/bin/env bash
# Minimal hard-coded supervisor for a torchrun job.
# Launch it DETACHED so it survives terminal close, e.g.:
#   nohup ./run_train.sh &     # background + ignores SIGHUP

set -euo pipefail  # exit on error/undefined var; robust scripting

# --------------------- HARD-CODED SETTINGS -----------------------------------
MAX_RETRIES=5            # How many times to retry after a non-zero exit.
                          # Set to -1 for infinite retries.
RETRY_DELAY=15            # Seconds to sleep between retries.

NPROC_PER_NODE=4          # Number of GPUs for torchrun on this machine.
CONFIG="configs/sscfg_imagenet_k8_train.py"  # Your training config path.

LOG_DIR="logs"            # Where to save logs.
LOG_FILE="$LOG_DIR/train_$(date +%Y%m%d_%H%M%S).log"  # Unique log per run.
# -----------------------------------------------------------------------------

mkdir -p "$LOG_DIR"
ln -sf "$(basename "$LOG_FILE")" "$LOG_DIR/latest.log"  # convenience symlink
echo "Logging to: $LOG_FILE"

attempt=0
while :; do
  attempt=$((attempt + 1))
  echo "[attempt $attempt] $(date '+%F %T') starting…" | tee -a "$LOG_FILE"

  # Run the job with line-buffered stdout/stderr for timely logs.
  # PYTHONUNBUFFERED avoids Python's stdio buffering in your training script.
  set +e
  PYTHONUNBUFFERED=1 stdbuf -oL -eL \
  torchrun --nnodes=1 --nproc_per_node="$NPROC_PER_NODE" \
           tools/train.py "$CONFIG" --launcher pytorch --diff_seed --no-validate \
           >>"$LOG_FILE" 2>&1
  rc=$?
  set -e

  echo "[attempt $attempt] $(date '+%F %T') exited with rc=$rc" | tee -a "$LOG_FILE"

  if [[ $rc -eq 0 ]]; then
    echo "[attempt $attempt] success; exiting." | tee -a "$LOG_FILE"
    exit 0
  fi

  # Stop retrying if we've hit the limit (unless MAX_RETRIES == -1).
  if [[ $MAX_RETRIES -ge 0 && $attempt -ge $MAX_RETRIES ]]; then
    echo "[attempt $attempt] reached MAX_RETRIES=$MAX_RETRIES; giving up." | tee -a "$LOG_FILE"
    exit "$rc"
  fi

  echo "[attempt $attempt] retrying in ${RETRY_DELAY}s…" | tee -a "$LOG_FILE"
  sleep "$RETRY_DELAY"
done
