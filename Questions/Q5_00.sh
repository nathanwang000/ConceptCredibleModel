#!/bin/bash

# this is Q5 for ccm eye/wl2 to try more settings

# Q5: relax assumption that S is contained in C; note the use of -s noise
# no need to rerun cbm assuming it didn't contain noise
# also remember to eval with -s noise

# ccm wl2
noise=1.0
track "python scripts/ccm.py --lr_step 15 --alpha 1e-4 -s noise -t $noise --n_shortcuts 10 --u_model_path outputs/relaxSinC$noise/87f4e850e36f11ebb773ac1f6b24a434/standard --c_model_path outputs/9843d41ae4c711ebb773ac1f6b24a434/concepts --reg wl2"

noise=0.9
track "python scripts/ccm.py --lr_step 15 --alpha 1e-4 -s noise -t $noise --n_shortcuts 10 --u_model_path outputs/relaxSinC$noise/8ab4e3541c9211ecb773ac1f6b24a434/standard --c_model_path outputs/9843d41ae4c711ebb773ac1f6b24a434/concepts --reg wl2"

noise=0.8
track "python scripts/ccm.py --lr_step 15 --alpha 1e-4 -s noise -t $noise --n_shortcuts 10 --u_model_path outputs/relaxSinC$noise/ed67ecbc1c3811ecb773ac1f6b24a434/standard --c_model_path outputs/9843d41ae4c711ebb773ac1f6b24a434/concepts --reg wl2"

noise=0.7
track "python scripts/ccm.py --lr_step 15 --alpha 1e-4 -s noise -t $noise --n_shortcuts 10 --u_model_path outputs/relaxSinC$noise/52c228961c8b11ecb773ac1f6b24a434/standard --c_model_path outputs/9843d41ae4c711ebb773ac1f6b24a434/concepts --reg wl2"

noise=0.6
track "python scripts/ccm.py --lr_step 15 --alpha 1e-4 -s noise -t $noise --n_shortcuts 10 --u_model_path outputs/relaxSinC$noise/672b313c1c3211ecb773ac1f6b24a434/standard --c_model_path outputs/9843d41ae4c711ebb773ac1f6b24a434/concepts --reg wl2"

noise=0.4
track "python scripts/ccm.py --lr_step 15 --alpha 1e-4 -s noise -t $noise --n_shortcuts 10 --u_model_path outputs/relaxSinC$noise/5efaf58c1c2411ecb773ac1f6b24a434/standard --c_model_path outputs/9843d41ae4c711ebb773ac1f6b24a434/concepts --reg wl2"

noise=0.2
track "python scripts/ccm.py --lr_step 15 --alpha 1e-4 -s noise -t $noise --n_shortcuts 10 --u_model_path outputs/relaxSinC$noise/48be12ae1c1a11ecb773ac1f6b24a434/standard --c_model_path outputs/9843d41ae4c711ebb773ac1f6b24a434/concepts --reg wl2"
