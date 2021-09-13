#!/bin/bash

# Q7: vary nSigma file; need to first run STD(X) before running STD(C, X), CCM EYE/RES
# because u_model_path need to be initialized to STD(X)

# for i in {1,5,20,30};
# do
#     # CBM
#     track  "python scripts/cbm.py --lr_step 15 -s outputs/aca36656e58e11ebb773ac1f6b24a434/cbm.pt -t 1 --n_shortcuts $i --c_model_path outputs/9843d41ae4c711ebb773ac1f6b24a434/concepts"

#     # STD(X)
#     track "python scripts/standard_model.py -s outputs/aca36656e58e11ebb773ac1f6b24a434/cbm.pt --n_shortcuts $i -t 1"
# done

############# n_shortcuts=1
# # STD(C, X) 
# track "python scripts/ccm.py --lr_step 15 --alpha 0 -s outputs/aca36656e58e11ebb773ac1f6b24a434/cbm.pt -t 1 --n_shortcuts 1 --u_model_path outputs/nSigmas/00042e18136e11ecb773ac1f6b24a434/standard --c_model_path outputs/9843d41ae4c711ebb773ac1f6b24a434/concepts --d_noise 0 --reg eye"

# # CCM EYE
# track "python scripts/ccm.py --lr_step 15 --alpha 1e-3 -s outputs/aca36656e58e11ebb773ac1f6b24a434/cbm.pt -t 1 --n_shortcuts 1 --u_model_path outputs/nSigmas/00042e18136e11ecb773ac1f6b24a434/standard --c_model_path outputs/9843d41ae4c711ebb773ac1f6b24a434/concepts --d_noise 0 --reg eye"

# # CCM RES
# track "python scripts/ccm_r.py --lr_step 15 -s outputs/aca36656e58e11ebb773ac1f6b24a434/cbm.pt -t 1 --n_shortcuts 1 --u_model_path outputs/nSigmas/00042e18136e11ecb773ac1f6b24a434/standard --c_model_path outputs/272d0b2a132511ecb773ac1f6b24a434/cbm"

############# n_shortcuts=5
# STD(C, X)
track "python scripts/ccm.py --lr_step 15 --alpha 0 -s outputs/aca36656e58e11ebb773ac1f6b24a434/cbm.pt -t 1 --n_shortcuts 5 --u_model_path outputs/nSigmas/a18625a6137911ecb773ac1f6b24a434/standard --c_model_path outputs/9843d41ae4c711ebb773ac1f6b24a434/concepts --d_noise 0 --reg eye"

# CCM EYE
track "python scripts/ccm.py --lr_step 15 --alpha 1e-3 -s outputs/aca36656e58e11ebb773ac1f6b24a434/cbm.pt -t 1 --n_shortcuts 5 --u_model_path outputs/nSigmas/a18625a6137911ecb773ac1f6b24a434/standard --c_model_path outputs/9843d41ae4c711ebb773ac1f6b24a434/concepts --d_noise 0 --reg eye"

# CCM RES
track "python scripts/ccm_r.py --lr_step 15 -s outputs/aca36656e58e11ebb773ac1f6b24a434/cbm.pt -t 1 --n_shortcuts 1 --u_model_path outputs/nSigmas/a18625a6137911ecb773ac1f6b24a434/standard --c_model_path outputs/906117a6132c11ecb773ac1f6b24a434/cbm"

############# n_shortcuts=20
# STD(C, X)
track "python scripts/ccm.py --lr_step 15 --alpha 0 -s outputs/aca36656e58e11ebb773ac1f6b24a434/cbm.pt -t 1 --n_shortcuts 20 --u_model_path outputs/nSigmas/30d45682138511ecb773ac1f6b24a434/standard --c_model_path outputs/9843d41ae4c711ebb773ac1f6b24a434/concepts --d_noise 0 --reg eye"

# CCM EYE
track "python scripts/ccm.py --lr_step 15 --alpha 1e-3 -s outputs/aca36656e58e11ebb773ac1f6b24a434/cbm.pt -t 1 --n_shortcuts 20 --u_model_path outputs/nSigmas/30d45682138511ecb773ac1f6b24a434/standard --c_model_path outputs/9843d41ae4c711ebb773ac1f6b24a434/concepts --d_noise 0 --reg eye"

# CCM RES
track "python scripts/ccm_r.py --lr_step 15 -s outputs/aca36656e58e11ebb773ac1f6b24a434/cbm.pt -t 1 --n_shortcuts 1 --u_model_path outputs/nSigmas/30d45682138511ecb773ac1f6b24a434/standard --c_model_path outputs/676b4e06133211ecb773ac1f6b24a434/cbm"

############# n_shortcuts=30
# STD(C, X)
track "python scripts/ccm.py --lr_step 15 --alpha 0 -s outputs/aca36656e58e11ebb773ac1f6b24a434/cbm.pt -t 1 --n_shortcuts 30 --u_model_path outputs/nSigmas/ffad38a6139011ecb773ac1f6b24a434/standard --c_model_path outputs/9843d41ae4c711ebb773ac1f6b24a434/concepts --d_noise 0 --reg eye"

# CCM EYE
track "python scripts/ccm.py --lr_step 15 --alpha 1e-3 -s outputs/aca36656e58e11ebb773ac1f6b24a434/cbm.pt -t 1 --n_shortcuts 30 --u_model_path outputs/nSigmas/ffad38a6139011ecb773ac1f6b24a434/standard --c_model_path outputs/9843d41ae4c711ebb773ac1f6b24a434/concepts --d_noise 0 --reg eye"

# CCM RES
track "python scripts/ccm_r.py --lr_step 15 -s outputs/aca36656e58e11ebb773ac1f6b24a434/cbm.pt -t 1 --n_shortcuts 1 --u_model_path outputs/nSigmas/ffad38a6139011ecb773ac1f6b24a434/standard --c_model_path outputs/30e590aa133a11ecb773ac1f6b24a434/cbm"
