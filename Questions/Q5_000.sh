#!/bin/bash

# this is Q5 for ccm res to try more regularizations: default wd 0.0004

# Q5: relax assumption that S is contained in C; note the use of -s noise
# no need to rerun cbm assuming it didn't contain noise
# also remember to eval with -s noise

noise=1.0
# track "python scripts/ccm_r.py --lr_step 15 -s noise -t $noise --n_shortcuts 10 --u_model_path outputs/relaxSinC1.0/87f4e850e36f11ebb773ac1f6b24a434/standard --c_model_path outputs/aca36656e58e11ebb773ac1f6b24a434/cbm --wd 0.004" # not working

track "python scripts/ccm_r.py --lr_step 15 -s noise -t $noise --n_shortcuts 10 --c_model_path outputs/aca36656e58e11ebb773ac1f6b24a434/cbm"

noise=0.9


noise=0.8


noise=0.7


noise=0.6


noise=0.4


noise=0.2

