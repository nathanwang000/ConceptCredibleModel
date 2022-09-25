#!/bin/bash

# Q5: relax assumption that S is contained in C; this setting is where S is outside of C but inside C and U
# no need to rerun std because swapping with noise for C doesn't affect std

noise=108

# # cbm
# track "python scripts/cbm.py --lr_step 15 -s outputs/aca36656e58e11ebb773ac1f6b24a434/cbm.pt -t 1 --n_shortcuts 10 --c_model_path outputs/9843d41ae4c711ebb773ac1f6b24a434/concepts --d_noise $noise"

# # std(c, x):
# track "python scripts/ccm.py --lr_step 15 --alpha 0 -s outputs/aca36656e58e11ebb773ac1f6b24a434/cbm.pt -t 1 --n_shortcuts 10 --u_model_path outputs/e19c89eaea4911ebb773ac1f6b24a434/standard --c_model_path outputs/9843d41ae4c711ebb773ac1f6b24a434/concepts --d_noise $noise"

# # ccm wl2
# track "python scripts/ccm.py --lr_step 15 --alpha 1e-3 -s outputs/aca36656e58e11ebb773ac1f6b24a434/cbm.pt -t 1 --n_shortcuts 10 --u_model_path outputs/e19c89eaea4911ebb773ac1f6b24a434/standard --c_model_path outputs/9843d41ae4c711ebb773ac1f6b24a434/concepts --d_noise $noise --reg wl2"

# ccm res: this need to wait for cbm run
track "python scripts/ccm_r.py --lr_step 15 -s outputs/aca36656e58e11ebb773ac1f6b24a434/cbm.pt -t 1 --n_shortcuts 10 --u_model_path outputs/e19c89eaea4911ebb773ac1f6b24a434/standard --c_model_path outputs/dnoise$noise/5c0385e61ca711ecb773ac1f6b24a434/cbm"



