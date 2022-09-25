#!/bin/bash

# Q4: relax assumption that S doesn't appear in C
# std: just use dnoise0, no need to rerun
# just very differnt c_models

###
# # concept: first run this
# for i in {0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9};
# do
#     track "python scripts/concept_model.py --transform flip --lr_step 1000 -t $i -s outputs/aca36656e58e11ebb773ac1f6b24a434/cbm.pt --n_shortcuts 10"
# done

# for i in {0.99,0.999};
# do
#     track "python scripts/concept_model.py --transform flip --lr_step 1000 -t $i -s outputs/aca36656e58e11ebb773ac1f6b24a434/cbm.pt --n_shortcuts 10"
# done

######## 1.0
# # cbm: need to finish before EYE RES
# track "python scripts/cbm.py --lr_step 15 -s outputs/aca36656e58e11ebb773ac1f6b24a434/cbm.pt -t 1 --c_model_path outputs/relaxCnoS1.0/0bb08d08180811ecb773ac1f6b24a434/concepts"

# # ccm eye
# track "python scripts/ccm.py --lr_step 15 --alpha 1e-3 -s outputs/aca36656e58e11ebb773ac1f6b24a434/cbm.pt -t 1 --n_shortcuts 10 --u_model_path outputs/e19c89eaea4911ebb773ac1f6b24a434/standard --c_model_path outputs/relaxCnoS1.0/0bb08d08180811ecb773ac1f6b24a434/concepts"

# # std(c,x)
# track "python scripts/ccm.py --lr_step 15 --alpha 0 -s outputs/aca36656e58e11ebb773ac1f6b24a434/cbm.pt -t 1 --n_shortcuts 10 --u_model_path outputs/e19c89eaea4911ebb773ac1f6b24a434/standard --c_model_path outputs/relaxCnoS1.0/0bb08d08180811ecb773ac1f6b24a434/concepts"

# # ccm_r: note use different cbm
# track "python scripts/ccm_r.py --lr_step 15 -s outputs/aca36656e58e11ebb773ac1f6b24a434/cbm.pt -t 1 --n_shortcuts 10 --u_model_path outputs/nSigmas/a18625a6137911ecb773ac1f6b24a434/standard --c_model_path outputs/relaxCnoS1.0/f7834c2c182911ecb773ac1f6b24a434/cbm"

######## 
# # cbm: need to finish before EYE RES; see Q4_cbm.sh

# # # ccm eye
# track "python scripts/ccm.py --lr_step 15 --alpha 1e-3 -s outputs/aca36656e58e11ebb773ac1f6b24a434/cbm.pt -t 1 --n_shortcuts 10 --u_model_path outputs/e19c89eaea4911ebb773ac1f6b24a434/standard --c_model_path outputs/relaxCnoS0.1/58ea6e54183a11ecb773ac1f6b24a434/concepts"

# track "python scripts/ccm.py --lr_step 15 --alpha 1e-3 -s outputs/aca36656e58e11ebb773ac1f6b24a434/cbm.pt -t 1 --n_shortcuts 10 --u_model_path outputs/e19c89eaea4911ebb773ac1f6b24a434/standard --c_model_path outputs/relaxCnoS0.2/58ea6e54183a11ecb773ac1f6b24a434/concepts"

# track "python scripts/ccm.py --lr_step 15 --alpha 1e-3 -s outputs/aca36656e58e11ebb773ac1f6b24a434/cbm.pt -t 1 --n_shortcuts 10 --u_model_path outputs/e19c89eaea4911ebb773ac1f6b24a434/standard --c_model_path outputs/relaxCnoS0.8/2f95d0a818bb11ecb773ac1f6b24a434/concepts"

# track "python scripts/ccm.py --lr_step 15 --alpha 1e-3 -s outputs/aca36656e58e11ebb773ac1f6b24a434/cbm.pt -t 1 --n_shortcuts 10 --u_model_path outputs/e19c89eaea4911ebb773ac1f6b24a434/standard --c_model_path outputs/relaxCnoS0.9/7657381a18cf11ecb773ac1f6b24a434/concepts"

# # # std(c,x)
# track "python scripts/ccm.py --lr_step 15 --alpha 0 -s outputs/aca36656e58e11ebb773ac1f6b24a434/cbm.pt -t 1 --n_shortcuts 10 --u_model_path outputs/e19c89eaea4911ebb773ac1f6b24a434/standard --c_model_path outputs/relaxCnoS0.1/58ea6e54183a11ecb773ac1f6b24a434/concepts"

# track "python scripts/ccm.py --lr_step 15 --alpha 0 -s outputs/aca36656e58e11ebb773ac1f6b24a434/cbm.pt -t 1 --n_shortcuts 10 --u_model_path outputs/e19c89eaea4911ebb773ac1f6b24a434/standard --c_model_path outputs/relaxCnoS0.1/58ea6e54183a11ecb773ac1f6b24a434/concepts"

track "python scripts/ccm.py --lr_step 15 --alpha 0 -s outputs/aca36656e58e11ebb773ac1f6b24a434/cbm.pt -t 1 --n_shortcuts 10 --u_model_path outputs/e19c89eaea4911ebb773ac1f6b24a434/standard --c_model_path outputs/relaxCnoS0.8/2f95d0a818bb11ecb773ac1f6b24a434/concepts"

track "python scripts/ccm.py --lr_step 15 --alpha 0 -s outputs/aca36656e58e11ebb773ac1f6b24a434/cbm.pt -t 1 --n_shortcuts 10 --u_model_path outputs/e19c89eaea4911ebb773ac1f6b24a434/standard --c_model_path outputs/relaxCnoS0.9/7657381a18cf11ecb773ac1f6b24a434/concepts"

# # # ccm_r: note use different cbm
# track "python scripts/ccm_r.py --lr_step 15 -s outputs/aca36656e58e11ebb773ac1f6b24a434/cbm.pt -t 1 --n_shortcuts 10 --u_model_path outputs/nSigmas/a18625a6137911ecb773ac1f6b24a434/standard --c_model_path outputs/relaxCnoS0.1/c0cba8f4189811ecb773ac1f6b24a434/cbm"

track "python scripts/ccm_r.py --lr_step 15 -s outputs/aca36656e58e11ebb773ac1f6b24a434/cbm.pt -t 1 --n_shortcuts 10 --u_model_path outputs/nSigmas/a18625a6137911ecb773ac1f6b24a434/standard --c_model_path outputs/relaxCnoS0.8/805a6ed818d211ecb773ac1f6b24a434/cbm"

track "python scripts/ccm_r.py --lr_step 15 -s outputs/aca36656e58e11ebb773ac1f6b24a434/cbm.pt -t 1 --n_shortcuts 10 --u_model_path outputs/nSigmas/a18625a6137911ecb773ac1f6b24a434/standard --c_model_path outputs/relaxCnoS0.9/209d7ea018ea11ecb773ac1f6b24a434/cbm"


