#!/bin/bash
#SBATCH --job-name=roinet_train
#SBATCH --account=kempner_rhakim_lab
#SBATCH --partition=kempner_requeue
#SBATCH --constraint=h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=374000M
#SBATCH --time=3-00:00:00
#SBATCH --requeue
#SBATCH --open-mode=append

## ---------------------------------------------------------------------------
## ROInet / SimCLR training worker.
##
## Generic worker file — runs both on SLURM and locally. The #SBATCH header is
## inert under non-sbatch invocations (bash comments). Override SLURM details
## (account, partition, constraint, time) at sbatch invocation time with the
## corresponding CLI flags rather than editing this file.
##
## Required env var:
##   PYTHON_EXEC  — absolute path to the conda env's python (e.g.
##                  /n/holylabs/.../envs/roicat2/bin/python).
##
## Positional args (all required):
##   $1  ANALYSIS_SCRIPT  absolute path to roicat/model_training/train_simclr.py
##   $2  RUN_NAME         short label, used for log lines + wandb display
##   $3  PATH_CONFIG      absolute path to the params JSON
##   $4  DIR_SAVE         directory for checkpoints, ONNX, logger.txt
## ---------------------------------------------------------------------------

set -u

ANALYSIS_SCRIPT="${1:-}"
RUN_NAME="${2:-}"
PATH_CONFIG="${3:-}"
DIR_SAVE="${4:-}"

if [ -z "$ANALYSIS_SCRIPT" ] || [ -z "$RUN_NAME" ] || [ -z "$PATH_CONFIG" ] || [ -z "$DIR_SAVE" ]; then
    echo "Usage: $0 <analysis_script> <run_name> <path_config> <dir_save>"
    echo "Required env: PYTHON_EXEC=/abs/path/to/conda-env/bin/python"
    exit 2
fi
[ -z "${PYTHON_EXEC:-}" ]   && { echo "ERROR: PYTHON_EXEC env var must point to the conda env's python"; exit 1; }
[ ! -x "$PYTHON_EXEC" ]     && { echo "ERROR: PYTHON_EXEC not executable: $PYTHON_EXEC"; exit 1; }
[ ! -f "$ANALYSIS_SCRIPT" ] && { echo "ERROR: analysis script not found: $ANALYSIS_SCRIPT"; exit 1; }
[ ! -f "$PATH_CONFIG" ]     && { echo "ERROR: config not found: $PATH_CONFIG"; exit 1; }

mkdir -p "$DIR_SAVE"

JOB_TAG="${SLURM_JOB_ID:-local}"
echo "============================================================"
echo "Run name:      $RUN_NAME"
echo "Job tag:       $JOB_TAG"
echo "Node:          ${SLURMD_NODENAME:-$(hostname)}"
echo "Start time:    $(date)"
echo "Python:        $PYTHON_EXEC"
echo "Config:        $PATH_CONFIG"
echo "Save dir:      $DIR_SAVE"
echo "============================================================"

"$PYTHON_EXEC" "$ANALYSIS_SCRIPT" \
    --directory_data "$(${PYTHON_EXEC} -c "import json; print(json.load(open('$PATH_CONFIG'))['data']['path_dataDir'])")" \
    --path_params "$PATH_CONFIG" \
    --directory_save "$DIR_SAVE"
RET=$?

echo "============================================================"
echo "Run $RUN_NAME (job=$JOB_TAG) exited with code: $RET"
echo "End time: $(date)"
echo "============================================================"
exit $RET
