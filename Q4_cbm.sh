#!/bin/bash

# Q4: relax assumption that S doesn't appear in C
# std: just use dnoise0, no need to rerun
# just very differnt c_models

# track "python scripts/cbm.py --lr_step 15 -s outputs/aca36656e58e11ebb773ac1f6b24a434/cbm.pt -t 1 --c_model_path outputs/relaxCnoS0.2/868efaf6184811ecb773ac1f6b24a434/concepts"

# track "python scripts/cbm.py --lr_step 15 -s outputs/aca36656e58e11ebb773ac1f6b24a434/cbm.pt -t 1 --c_model_path outputs/relaxCnoS0.3/70081722185c11ecb773ac1f6b24a434/concepts"

# track "python scripts/cbm.py --lr_step 15 -s outputs/aca36656e58e11ebb773ac1f6b24a434/cbm.pt -t 1 --c_model_path outputs/relaxCnoS0.4/c1eecc1e186d11ecb773ac1f6b24a434/concepts"

# track "python scripts/cbm.py --lr_step 15 -s outputs/aca36656e58e11ebb773ac1f6b24a434/cbm.pt -t 1 --c_model_path outputs/relaxCnoS0.5/6db64f00188111ecb773ac1f6b24a434/concepts"

track "python scripts/cbm.py --lr_step 15 -s outputs/aca36656e58e11ebb773ac1f6b24a434/cbm.pt -t 1 --c_model_path outputs/relaxCnoS0.9/7657381a18cf11ecb773ac1f6b24a434/concepts"

# track "python scripts/cbm.py --lr_step 15 -s outputs/aca36656e58e11ebb773ac1f6b24a434/cbm.pt -t 1 --c_model_path outputs/relaxCnoS0.8/2f95d0a818bb11ecb773ac1f6b24a434/concepts"

# track "python scripts/cbm.py --lr_step 15 -s outputs/aca36656e58e11ebb773ac1f6b24a434/cbm.pt -t 1 --c_model_path outputs/relaxCnoS0.7/486e270618a711ecb773ac1f6b24a434/concepts"

# track "python scripts/cbm.py --lr_step 15 -s outputs/aca36656e58e11ebb773ac1f6b24a434/cbm.pt -t 1 --c_model_path outputs/relaxCnoS0.6/ebe01330189411ecb773ac1f6b24a434/concepts"
