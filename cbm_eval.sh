#!/bin/bash
# eg. bash cbm_eval.sh outputs/...
echo $1
echo "clean acc"
python scripts/cbm.py --eval -s clean -o $1
echo "t=0 acc"
python scripts/cbm.py --eval -s noise -t 0 -o $1
echo "t=1 acc"
python scripts/cbm.py --eval -s outputs/aca36656e58e11ebb773ac1f6b24a434/cbm.pt -t 1 -o $1
