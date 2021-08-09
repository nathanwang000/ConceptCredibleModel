'''
file for saving different u_models
'''

import sys, os
import tqdm
import numpy as np
import seaborn as sns
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from torchvision import transforms
from IPython.display import Image as showImg
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Subset
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from copy import deepcopy
import matplotlib
import pandas as pd
from sklearn.model_selection import train_test_split
import argparse
from torch.optim import lr_scheduler
import torch.nn.functional as F
from functools import partial

###
FilePath = os.path.dirname(os.path.abspath(__file__))
RootPath = os.path.dirname(FilePath)
if RootPath not in sys.path: # parent directory
    sys.path = [RootPath] + sys.path
from lib.models import MLP, LambdaNet, CCM, ConcatNet, CUB_Subset_Concept_Model, CCM_res
from lib.models import CUB_Noise_Concept_Model
from lib.data import small_CUB, CUB, SubColumn, CUB_train_transform, CUB_test_transform
from lib.data import SubAttr
from lib.train import train
from lib.eval import get_output, test, plot_log, shap_net_x, shap_ccm_c, bootstrap
from lib.utils import birdfile2class, birdfile2idx, is_test_bird_idx, get_bird_bbox, get_bird_class, get_bird_part, get_part_location, get_multi_part_location, get_bird_name
from lib.utils import get_attribute_name, code2certainty, get_class_attributes, get_image_attributes, describe_bird
from lib.utils import get_attr_names, dfs_freeze
from lib.regularization import EYE

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_type", choices=['imagenet', 'fresh'])
    parser.add_argument("-o", "--outputs_dir", default=f"outputs",
                        help="where to save all the outputs")

    args = parser.parse_args()
    print(args)
    return args


if __name__ == '__main__':
    flags = get_args()

    model_name = f"{RootPath}/{flags.outputs_dir}/u_model_{flags.model_type}"
    device = 'cuda'
    print(model_name)

    pretrained = (flags.model_type == 'imagenet')
    net = torch.hub.load('pytorch/vision:v0.9.0', 'inception_v3', pretrained=pretrained)
    net.fc = nn.Linear(2048, 200)
    net.AuxLogits.fc = nn.Linear(768, 200)
    net.to(device)

    torch.save(net, model_name + ".pt")

    
