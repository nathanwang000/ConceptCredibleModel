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
from lib.models import MLP, LambdaNet, CCM, ConcatNet
from lib.data import MIMIC, SubColumn, MIMIC_train_transform, MIMIC_test_transform, subsample_mimic
from lib.train import train
from lib.eval import get_output
from lib.regularization import EYE, wL2

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_name", required=True,
                        help="what is the model name w/o postfix")
    parser.add_argument("--task", default="Edema",
                        help="which task to train concept model")
    parser.add_argument("--seed", type=int, default=42,
                        help="seed for reproducibility")
        
    args = parser.parse_args()
    print(args)
    return args

if __name__ == '__main__':
    flags = get_args()
    model_name = f"{RootPath}/{flags.model_name}"
    print(model_name)

    task = flags.task
    mimic = MIMIC_test_transform(MIMIC(task))
    
    subcolumn = lambda d: SubColumn(d, ['x', 'y'])
    load = lambda d: DataLoader(subcolumn(d), batch_size=32,
                                shuffle=False, num_workers=8,
                                drop_last=False)
    loader = load(mimic)
    net = torch.load(f'{model_name}.pt')
    o = get_output(net, loader)
    torch.save(o, f'{model_name}_{flags.task}.npz')

    
    
