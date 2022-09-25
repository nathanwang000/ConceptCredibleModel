#!/bin/bash

# Q5: relax assumption that S is contained in C; note the use of -s noise; this setting is for S is outside of both C and U
# no need to rerun cbm assuming it didn't contain noise
# also remember to eval with -s noise

# std(x): 1.0: already ran outputs/relaxSinC1.0/87f4e850e36f11ebb773ac1f6b24a434/standard
for i in {0.7,0.9}; #{0.2,0.4,0.6,0.8};
do
    track "python scripts/standard_model.py -s noise --n_shortcuts 10 -t $i"
done

# # std(c, x): note to change u_model_path for different std
# track "python scripts/ccm.py --lr_step 15 --alpha 0 -s noise -t 1 --n_shortcuts 10 --u_model_path outputs/87f4e850e36f11ebb773ac1f6b24a434/standard --c_model_path outputs/9843d41ae4c711ebb773ac1f6b24a434/concepts"

# # ccm res: change u_model_path and c_model_path to standard model and regular cbm
# track "python scripts/ccm_r.py --lr_step 15 -s noise -t 1 --n_shortcuts 10 --u_model_path outputs/87f4e850e36f11ebb773ac1f6b24a434/standard --c_model_path outputs/aca36656e58e11ebb773ac1f6b24a434/cbm"

# # ccm wl2
# track "python scripts/ccm.py --lr_step 15 --alpha 1e-3 -s noise -t 1 --n_shortcuts 10 --u_model_path outputs/87f4e850e36f11ebb773ac1f6b24a434/standard --c_model_path outputs/9843d41ae4c711ebb773ac1f6b24a434/concepts --reg wl2"



