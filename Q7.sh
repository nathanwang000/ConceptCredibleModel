#!/bin/bash

# Q7: vary nSigma file; need to first run STD(X) before running STD(C, X), CCM EYE/RES
# because u_model_path need to be initialized to STD(X)
for i in {1,5,20,30};
do
    track "python scripts/standard_model.py -s outputs/aca36656e58e11ebb773ac1f6b24a434/cbm.pt --n_shortcuts $i -t 1"
    # track "python scripts/ccm.py --lr_step 15 --alpha 0 -s outputs/aca36656e58e11ebb773ac1f6b24a434/cbm.pt -t 1 --n_shortcuts $i --u_model_path outputs/e19c89eaea4911ebb773ac1f6b24a434/standard --c_model_path outputs/9843d41ae4c711ebb773ac1f6b24a434/concepts --d_noise 0 --reg eye"
    # track  "python scripts/cbm.py --lr_step 15 -s outputs/aca36656e58e11ebb773ac1f6b24a434/cbm.pt -t 1 --n_shortcuts $i --c_model_path outputs/9843d41ae4c711ebb773ac1f6b24a434/concepts"
done
