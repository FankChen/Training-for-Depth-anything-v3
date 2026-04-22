#!/bin/bash
# Record the first run (job 12200409, lr=1e-4, 25 epochs, H200 reservation) into
# experiments.xlsx. Run this after the job has finished and test_metrics.json
# exists in logs/depth_anything_v3/kitti_eigen_vitl/.
set -e

PROJ=$(cd "$(dirname "$0")/.." && pwd)
cd "${PROJ}"

PROJECT_CODE="BH-000425-08-05"
RUN_NAME="${PROJECT_CODE}-VDA-V3-KITTI-lr1e-4-run1"
LOG_DIR="${PROJ}/logs/depth_anything_v3/kitti_eigen_vitl"
METRICS_JSON="${LOG_DIR}/test_metrics.json"

if [ ! -f "${METRICS_JSON}" ]; then
    echo "WARN: ${METRICS_JSON} not found yet — job still running?"
fi

BEST_CKPT=""
if [ -f "${METRICS_JSON}" ]; then
    BEST_CKPT=$(python -c "import json; print(json.load(open('${METRICS_JSON}')).get('_best_ckpt',''))" 2>/dev/null || true)
fi

python scripts/record_run.py \
    --xlsx experiments.xlsx \
    --run-name "${RUN_NAME}" \
    --project-code "${PROJECT_CODE}" \
    --stage train --status done \
    --job-id "12200409" \
    --submitted-at "2026-04-22 10:26:00" \
    --finished-at "$(date '+%Y-%m-%d %H:%M:%S')" \
    --encoder vitl --freeze-backbone true --freeze-until 5 \
    --lr "1e-4" --backbone-lr-factor 0.1 --batch-size 4 \
    --max-epochs 25 --precision 16-mixed \
    --input-hw "350x1218" --dataset-split "eigen_train/eigen_test" \
    --ckpt "${BEST_CKPT}" \
    --log-dir "${LOG_DIR}" \
    --metrics-json "${METRICS_JSON}" \
    --notes "First successful run on H200 reservation (rng-dl01-w24n01). lr=1e-4." \
    --update
