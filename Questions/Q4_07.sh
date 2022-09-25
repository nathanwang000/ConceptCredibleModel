#!/bin/bash

## Q4 for 0.7
# ccm eye
track "python scripts/ccm.py --lr_step 15 --alpha 1e-3 -s outputs/aca36656e58e11ebb773ac1f6b24a434/cbm.pt -t 1 --n_shortcuts 10 --u_model_path outputs/e19c89eaea4911ebb773ac1f6b24a434/standard --c_model_path outputs/relaxCnoS0.7/486e270618a711ecb773ac1f6b24a434/concepts"

# std(c, x)
track "python scripts/ccm.py --lr_step 15 --alpha 0 -s outputs/aca36656e58e11ebb773ac1f6b24a434/cbm.pt -t 1 --n_shortcuts 10 --u_model_path outputs/e19c89eaea4911ebb773ac1f6b24a434/standard --c_model_path outputs/relaxCnoS0.7/486e270618a711ecb773ac1f6b24a434/concepts"

# ccmr
track "python scripts/ccm_r.py --lr_step 15 -s outputs/aca36656e58e11ebb773ac1f6b24a434/cbm.pt -t 1 --n_shortcuts 10 --u_model_path outputs/nSigmas/a18625a6137911ecb773ac1f6b24a434/standard --c_model_path outputs/relaxCnoS0.7/35588abc18d911ecb773ac1f6b24a434/cbm"
