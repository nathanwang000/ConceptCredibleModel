#!/bin/bash
# eg. bash ccm_eval.sh outputs/...
echo "clean acc"
python scripts/ccm.py --eval -s clean -o $1
echo "t=0 acc"
python scripts/ccm.py --eval -s noise -t 0 -o $1
echo "t=1 acc"
python scripts/ccm.py --eval -s outputs/aca36656e58e11ebb773ac1f6b24a434/cbm.pt -t 1 -o $1

