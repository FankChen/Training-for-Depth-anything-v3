#!/bin/bash
# Launch an interactive GPU shell on the cluster for debugging.
# Usage:  bash scripts/interactive.sh
exec bsub -Is -q inter_a100 -n 4 -M 15000 -gpu 'num=1' -W 8:00 /bin/bash
