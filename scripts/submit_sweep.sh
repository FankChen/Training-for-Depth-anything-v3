#!/bin/bash
# -----------------------------------------------------------------------------
# Submit a learning-rate sweep for VDA-V3 KITTI fine-tuning on the PUBLIC queue
# (batch_b200, no -U, no -m). Each submission is independently named with the
# Bosch project code prefix.
#
# Usage:
#   ./scripts/submit_sweep.sh                # submits the default 5 LRs
#   LRS="1e-5 5e-6" ./scripts/submit_sweep.sh   # custom list
#   MAX_EPOCHS=15 LRS="3e-5" ./scripts/submit_sweep.sh
# -----------------------------------------------------------------------------
set -e

PROJ=$(cd "$(dirname "$0")/.." && pwd)
cd "${PROJ}"

PROJECT_CODE="${PROJECT_CODE:-BH-000425-08-05}"
MAX_EPOCHS="${MAX_EPOCHS:-25}"
# Small LRs on purpose — DA3 pretrained weights are already strong.
LRS="${LRS:-5e-5 3e-5 1e-5 5e-6 1e-6}"

mkdir -p jobs

echo "Submitting sweep on PUBLIC queue batch_b200 (no reservation)"
echo "  LRS       = ${LRS}"
echo "  EPOCHS    = ${MAX_EPOCHS}"
echo "  PROJECT   = ${PROJECT_CODE}"
echo

for LR in ${LRS}; do
    TAG="lr${LR}"
    RUN_NAME="${PROJECT_CODE}-VDA-V3-KITTI-${TAG}"
    NOTES="LR sweep (frozen 5 ep then unfreeze, cosine, warmup 2). lr=${LR}"

    echo "--- submitting ${RUN_NAME} ---"
    # Export so `bsub -env "all"` forwards them to the job (avoids fragile
    # comma-separated -env list which breaks on spaces/commas in NOTES).
    export TAG LR MAX_EPOCHS PROJECT_CODE NOTES
    bsub -J "${RUN_NAME}" -env "all" < scripts/train_sweep.bsub
done

echo
echo "Done. Check with: bjobs -a | grep VDA-V3-KITTI"
