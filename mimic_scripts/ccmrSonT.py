'''
this files trains an sequential or independent CCM with residual
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
from lib.models import MLP, LambdaNet, CCM_res
from lib.data import MIMIC, SubColumn, MIMIC_train_transform, MIMIC_test_transform, subsample_mimic
from lib.train import train
from lib.eval import get_output, test_auc
from lib.regularization import EYE

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--outputs_dir", default=f"outputs",
                        help="where to save all the outputs")
    parser.add_argument("--task", default="Pneumonia",
                        help="which task to train concept model")
    parser.add_argument("--eval", action="store_true",
                        help="whether or not to eval the learned model")
    parser.add_argument("--alpha", default=0, type=float,
                        help="regularization strength for EYE")
    parser.add_argument("--retrain", action="store_true",
                        help="retrain using all train val data")
    parser.add_argument("--seed", type=int, default=42,
                        help="seed for reproducibility")
    parser.add_argument("--transform", default="cbm",
                        help="transform mode to use")
    parser.add_argument("--wd", default=0.0004, type=float,
                        help="weight decay for the model")
    parser.add_argument("--d_noise", default=0, type=int,
                        help="wrong expert dimensions (noise dimensions in c)")
    parser.add_argument("--lr_step", type=int, default=1000,
                        help="learning rate decay steps")
    parser.add_argument("--n_epochs", type=int, default=100,
                        help="max number of epochs to train")
    parser.add_argument("--use_aux", action="store_true",
                        help="auxilliary loss for inception")
    parser.add_argument("--c_model_path", type=str,
                        default="gold_models/flip/concepts",
                        help="concept model path starting from root (ignore .pt)")
    parser.add_argument("--u_model_path", type=str,
                        default=None, # "gold_models/crop/standard",
                        help="unknown concept model path starting from root (ignore .pt), if None then train from scratch")
    parser.add_argument("--concept_path", type=str,
                        default="outputs/concepts/concepts_108.txt",
                        help="path to file containing concept names")
    # shortcut related
    parser.add_argument("-s", "--shortcut", default="clean",
                        help="shortcut transform to use; clean: no shortcut; noise: shortcut dependent on y; else: shortcut dependent on yhat computed from the model path")
    parser.add_argument("-t", "--threshold", default=1.0, type=float,
                        help="shortcut threshold to use (1 always Y dependent, 0 ind)")
    parser.add_argument("--n_shortcuts", default=10, type=int,
                        help="number of shortcuts")
    parser.add_argument("--subsample", default="", type=str,
                        help="field to subsample e.g., gender")
        
    args = parser.parse_args()
    print(args)
    return args

def ccm(flags, concept_model_path,
        loader_xy, loader_xy_eval, loader_xy_te, loader_xy_val=None,
        net_s=None,
        n_epochs=10, report_every=1, lr_step=1000,
        u_model_path=None,
        alpha=0, # regularizaiton strength
        device='cuda', savepath=None, use_aux=False):
    '''
    loader_xy_eval is the evaluation of loader_xy
    if loader_xy_val: use early stopping, otherwise train for the number of epochs
    '''

    # known concept model: note here is a cbm model
    cbm = torch.load(f'{RootPath}/{concept_model_path}.pt')

    # unknown concept model
    if u_model_path:
        x2u = torch.load(f'{RootPath}/{u_model_path}.pt')
        x2u.aux_logits = False
    else:
        x2u = torch.hub.load('pytorch/vision:v0.9.0', 'inception_v3', pretrained=True)
        x2u.fc = nn.Linear(2048, 2)
        x2u.aux_logits = False
        # x2u.AuxLogits.fc = nn.Linear(768, 2)

    # combined model
    net = CCM_res(cbm, x2u)
    net.to(device)

    # print('task auc before training: {:.1f}%'.format(
    #     run_test(net, loader_xy_te) * 100))
    
    criterion = lambda o_y, y: F.cross_entropy(o_y, y)
    
    # train
    opt = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=flags.wd)
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
        # shortcut specific done
        n_epochs=n_epochs, report_every=report_every,
        device=device, savepath=savepath,
        scheduler=scheduler, **kwargs)

    if loader_xy_val:
        log  = run_train(
            report_dict={'val auc': (lambda m: run_test(m, loader_xy_val) * 100, 'max'),
                         'train auc': (lambda m: run_test(m, loader_xy_eval) * 100,
                                       'max')},
                    early_stop_metric='val auc')
    else:
        log = run_train(
            report_dict={'test auc': (lambda m: run_test(m, loader_xy_te) * 100, 'max'),
                         'train auc': (lambda m: run_test(m, loader_xy_eval) * 100,
                                       'max')})

    print('task auc after training: {:.1f}%'.format(run_test(net, loader_xy_te) * 100))
    return net

if __name__ == '__main__':
    flags = get_args()
    model_name = f"{RootPath}/{flags.outputs_dir}/ccmr"
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
    # train indices already are used for training CBM, so we split validation in half to train the residual model
    train_indices, val_indices = train_test_split(val_indices, test_size=0.5,
                                                  random_state=flags.seed)
    
    
    # define dataloader: mimic_train_eval is used to evaluate training data
    mimic_train = MIMIC_train_transform(Subset(mimic, train_indices), mode=flags.transform)
    mimic_val = MIMIC_test_transform(Subset(mimic, val_indices),  mode=flags.transform)
    mimic_test = MIMIC_test_transform(Subset(mimic, test_indices), mode=flags.transform)
    mimic_train_eval = MIMIC_test_transform(Subset(mimic, train_indices), mode=flags.transform)

    # dataset, note only x, y not xy
    subcolumn = lambda d: SubColumn(d, ['x', 'y'])
    load = lambda d, shuffle: DataLoader(subcolumn(d), batch_size=32,
                                         shuffle=shuffle, num_workers=8,
                                         drop_last=True)
    loader_xy = load(mimic_train, True)
    loader_xy_val = load(mimic_val, False)
    loader_xy_te = load(mimic_test, False)
    loader_xy_eval = load(mimic_train_eval, False)
    
    print(f"# train: {len(mimic_train)}, # val: {len(mimic_val)}, # test: {len(mimic_test)}")

    run_train = lambda **kwargs:ccm(
        flags, flags.c_model_path,
        loader_xy, loader_xy_eval,
        loader_xy_te, net_s=net_s,
        alpha=flags.alpha,
        u_model_path = flags.u_model_path,
        n_epochs=flags.n_epochs, report_every=1,
        lr_step=flags.lr_step,
        savepath=model_name, use_aux=flags.use_aux, **kwargs)
    run_test = partial(test_auc, 
                       device='cuda',
                       max_batches=None if flags.eval else 100,
                       # shortcut specific
                       shortcut_mode = flags.shortcut,
                       shortcut_threshold = flags.threshold,
                       n_shortcuts = flags.n_shortcuts,
                       shortcut_subsample = flags.subsample,
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
        
