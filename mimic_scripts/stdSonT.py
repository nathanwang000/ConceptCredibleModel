'''
this file trains a standard model to either train (x, y) or (x, shortcut)
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

DEVICE = 'cuda'

###
FilePath = os.path.dirname(os.path.abspath(__file__))
RootPath = os.path.dirname(FilePath)
if RootPath not in sys.path: # parent directory
    sys.path = [RootPath] + sys.path
from lib.models import MLP
from lib.data import MIMIC, SubColumn, MIMIC_train_transform, MIMIC_test_transform
from lib.data import MIMIC, subsample_mimic
from lib.train import train
from lib.eval import get_output, test_auc
from lib.utils import birdfile2class, birdfile2idx, is_test_bird_idx, describe_bird
from lib.utils import get_bird_bbox, get_bird_class, get_bird_part, get_part_location
from lib.utils import get_multi_part_location, get_bird_name, get_image_attributes
from lib.utils import get_attribute_name, code2certainty, get_class_attributes

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--outputs_dir", default=f"outputs",
                        help="where to save all the outputs")
    parser.add_argument("--eval", action="store_true",
                        help="whether or not to eval the learned model")
    parser.add_argument("--task", default="Pneumonia",
                        help="which task to train concept model")
    parser.add_argument("--retrain", action="store_true",
                        help="retrain using all train val data")
    parser.add_argument("--saveall", action="store_true",
                        help="save all epochs")
    parser.add_argument("--seed", type=int, default=42,
                        help="seed for reproducibility")
    parser.add_argument("--transform", default="cbm",
                        help="transform mode to use")
    parser.add_argument("--name", default="standard",
                        help="model name to save")
    # shortcut related
    parser.add_argument("--predict_shortcut", action="store_true",
                        help="do shortcut prediction (o.w. predict y)")
    parser.add_argument("-s", "--shortcut", default="clean",
                        help="shortcut transform to use; clean: no shortcut; noise: shortcut dependent on y; else: shortcut dependent on yhat computed from the model path")
    parser.add_argument("-t", "--threshold", default=1.0, type=float,
                        help="shortcut threshold to use (1 always Y dependent, 0 ind)")
    parser.add_argument("--smax", default=0.1, type=float,
                        help="maximum noise added")
    parser.add_argument("--n_shortcuts", default=2, type=int,
                        help="number of shortcuts")
    parser.add_argument("--subsample", default="", type=str,
                        help="field to subsample e.g., gender")
    parser.add_argument("--init_model_path", type=str, default="",
                        help="model intialization, if None then train from scratch")
    # other training stuff
    parser.add_argument("--lr_step", type=int, default=15,
                        help="learning rate decay steps")
    parser.add_argument("--n_epochs", type=int, default=100,
                        help="max number of epochs to train")
    parser.add_argument("--use_aux", action="store_true",
                        help="auxilliary loss for inception")
    
    args = parser.parse_args()
    print(args)
    return args

def standard_model(flags, loader_xy, loader_xy_eval, loader_xy_te, loader_xy_val=None,
                   n_epochs=10, report_every=1, lr_step=1000, net_s=None,
                   device=DEVICE, savepath=None, use_aux=False):
    '''
    loader_xy_eval is the evaluation of loader_xy
    if loader_xy_val: use early stopping, otherwise train for the number of epochs
    '''
    # regular model
    if len(flags.init_model_path) != 0:
        net = torch.load(flags.init_model_path)
    else:
        net = torch.hub.load('pytorch/vision:v0.9.0',
                             'inception_v3', pretrained=True)
        net.fc = nn.Linear(2048, 2) # mimic dataset is binary
        net.aux_logits = False
        # net.AuxLogits.fc = nn.Linear(768, 200)
        net.to(device)

    criterion = lambda o, y: F.cross_entropy(o, y)
    
    # train
    opt = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0004)
    scheduler = lr_scheduler.StepLR(opt, step_size=lr_step)

    run_train = lambda **kwargs: train(
        net, loader_xy, opt, criterion=criterion,
        max_batches=300, # so that see progress faster                        
        # shortcut specific
        shortcut_mode = flags.shortcut,
        shortcut_threshold = flags.threshold,
        n_shortcuts = flags.n_shortcuts,
        shortcut_subsample = flags.subsample,
        net_shortcut = net_s,
        smax = flags.smax,
        # shortcut specific done
        n_epochs=n_epochs, report_every=report_every,
        device=device, savepath=savepath,
        saveall=flags.saveall,
        scheduler=scheduler, **kwargs)
    if loader_xy_val:
        log = run_train(
            report_dict={'val auc': (lambda m: run_test(m, loader_xy_val) * 100, 'max'),
                         'train auc': (lambda m: run_test(m,
                                                          loader_xy_eval) * 100, 'max')},
            early_stop_metric='val auc')
    else:
        log = run_train(
            report_dict={'test auc': (lambda m: run_test(m, loader_xy_te) * 100, 'max'),
                         'train auc': (lambda m: run_test(m,
                                                          loader_xy_eval) * 100, 'max')})

    print('task auc after training: {:.1f}%'.format(run_test(net, loader_xy_te) * 100))
    return net

if __name__ == '__main__':
    flags = get_args()
    model_name = f"{RootPath}/{flags.outputs_dir}/{flags.name}"
    print(model_name)

    task = flags.task
    mimic = MIMIC(task)
    # subsample the data
    if flags.shortcut not in ['clean', 'noise']:
        net_s = torch.load(flags.shortcut)
    else:
        net_s = None

    if flags.subsample and flags.shortcut != 'clean':
        # subsample "subsample" field; only work for binary field
        print(f"subsampling {flags.subsample}")
        mimic = subsample_mimic(mimic,
                                field=flags.subsample,
                                threshold=flags.threshold,
                                net_s=net_s)
    
    indices = list(range(len(mimic)))
    labels = list(mimic.df[task])

    test_ratio = 0.2
    train_val_indices, test_indices, train_val_labels, _ = train_test_split(
        indices, labels, test_size=test_ratio, stratify=labels, random_state=flags.seed)
    val_ratio = 0.2

    train_indices, val_indices = train_test_split(train_val_indices, test_size=val_ratio,
                                                  stratify=train_val_labels,
                                                  random_state=flags.seed)

    # define dataloader: mimic_train_eval is used to evaluate training data
    mimic_train = MIMIC_train_transform(Subset(mimic, train_indices), mode=flags.transform)
    mimic_val = MIMIC_test_transform(Subset(mimic, val_indices),  mode=flags.transform)
    mimic_test = MIMIC_test_transform(Subset(mimic, test_indices), mode=flags.transform)
    mimic_train_eval = MIMIC_test_transform(Subset(mimic, train_indices), mode=flags.transform)

    # dataset    
    if flags.predict_shortcut:
        subcolumn = lambda d: SubColumn(d, ['x', 's'])
    else:
        subcolumn = lambda d: SubColumn(d, ['x', 'y'])

    load = lambda d, shuffle: DataLoader(subcolumn(d), batch_size=32,
                                shuffle=shuffle, num_workers=8)
    loader_xy = load(mimic_train, True)
    loader_xy_val = load(mimic_val, False)
    loader_xy_te = load(mimic_test, False)
    loader_xy_eval = load(mimic_train_eval, False)
    
    print(f"# train: {len(mimic_train)}, # val: {len(mimic_val)}, # test: {len(mimic_test)}")

    run_train = lambda **kwargs: standard_model(
        flags, loader_xy, loader_xy_eval,
        loader_xy_te, net_s=net_s,
        n_epochs=flags.n_epochs, report_every=1,
        lr_step=flags.lr_step,
        savepath=model_name, use_aux=flags.use_aux, **kwargs)
    run_test = partial(test_auc, device=DEVICE,
                       max_batches= None if flags.eval else 100,
                       # shortcut specific
                       shortcut_mode = flags.shortcut,
                       shortcut_threshold = flags.threshold,
                       n_shortcuts = flags.n_shortcuts,
                       shortcut_subsample = flags.subsample,
                       smax = flags.smax,                       
                       net_shortcut = net_s)
    
    if flags.eval:
        l, s, r = run_test(torch.load(f'{model_name}.pt'), loader_xy_te)
        print(f'task auc after training: ({l*100:.1f}, {s*100:.1f}, {r*100:.1f})')
        
    elif flags.retrain:
        mimic_train = MIMIC_train_transform(Subset(mimic, train_val_indices),
                                        mode=flags.transform)
        mimic_train_eval = MIMIC_test_transform(Subset(mimic, train_val_indices),
                                        mode=flags.transform)
        loader_xy = load(mimic_train, True)
        loader_xy_eval = load(mimic_train_eval, False)

        net = run_train()
    else:
        net = run_train(loader_xy_val=loader_xy_val)
        
