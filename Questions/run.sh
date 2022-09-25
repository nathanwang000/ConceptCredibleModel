#!/bin/bash

# just a tmp file to run jobs
echo "n_shortcuts $1"

for i in {1..6};
do
    track "python scripts/ccm.py --lr_step 15 --alpha 1e-$i -s outputs/aca36656e58e11ebb773ac1f6b24a434/cbm.pt -t 1 --n_shortcuts $1 --u_model_path outputs/e19c89eaea4911ebb773ac1f6b24a434/standard --c_model_path outputs/9843d41ae4c711ebb773ac1f6b24a434/concepts --d_noise 0 --reg eye"
done
