#!/bin/bash

noise=0.6
# Q5: relax assumption that S is contained in C; note the use of -s noise
# no need to rerun cbm assuming it didn't contain noise
# also remember to eval with -s noise

# std(x): already ran in Q5_2.sh

# ccm res: change u_model_path and c_model_path to standard model and regular cbm
track "python scripts/ccm_r.py --lr_step 15 -s noise -t $noise --n_shortcuts 10 --u_model_path outputs/relaxSinC$noise/672b313c1c3211ecb773ac1f6b24a434/standard --c_model_path outputs/aca36656e58e11ebb773ac1f6b24a434/cbm"

# std(c, x): note to change u_model_path for different std
track "python scripts/ccm.py --lr_step 15 --alpha 0 -s noise -t $noise --n_shortcuts 10 --u_model_path outputs/relaxSinC$noise/672b313c1c3211ecb773ac1f6b24a434/standard --c_model_path outputs/9843d41ae4c711ebb773ac1f6b24a434/concepts"

# ccm wl2
track "python scripts/ccm.py --lr_step 15 --alpha 1e-3 -s noise -t $noise --n_shortcuts 10 --u_model_path outputs/relaxSinC$noise/672b313c1c3211ecb773ac1f6b24a434/standard --c_model_path outputs/9843d41ae4c711ebb773ac1f6b24a434/concepts --reg wl2"


