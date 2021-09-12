#!/bin/bash

# Q5: relax assumption that S is contained in C: add_s
for i in {0,30,70,100};
do
    track  "python scripts/cbm.py --lr_step 15 -s outputs/aca36656e58e11ebb773ac1f6b24a434/cbm.pt -t 1 --c_model_path outputs/9843d41ae4c711ebb773ac1f6b24a434/concepts --add_s --d_noise $i"
done
