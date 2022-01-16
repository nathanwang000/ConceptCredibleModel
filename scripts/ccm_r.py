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
    parser.add_argument("-o", "--outputs_dir", default=f"outputs",
                        help="where to save all the outputs")
    parser.add_argument("--eval", action="store_true",
                        help="whether or not to eval the learned model")
    parser.add_argument("--split_val", action="store_true",
                        help="whether or not to split the validation set (used for ccmr to not reuse training)")
    parser.add_argument("--ind", action="store_true",
                        help="whether or not to train independent CCM")
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
    
        
    args = parser.parse_args()
    print(args)
    return args

def ccm(flags, attr_names, concept_model_path,
        loader_xy, loader_xy_eval, loader_xy_te, loader_xy_val=None,
        independent=False, net_s=None,
        n_epochs=10, report_every=1, lr_step=1000,
        u_model_path=None,
        alpha=0, # regularizaiton strength
        device='cuda', savepath=None, use_aux=False):
    '''
    loader_xy_eval is the evaluation of loader_xy
    if loader_xy_val: use early stopping, otherwise train for the number of epochs
    
    independent: whether or not to train ccm independently
    '''
    attr_full_names = get_attr_names(f"{RootPath}/outputs/concepts/concepts_108.txt")
    assert len(attr_full_names) == 108, "108 features required"
    # use subset of attributes: don't need transition b/c it was jointly trained    
    transition = CUB_Subset_Concept_Model(attr_names, attr_full_names)
    
    d_x2u = 200 # give it a chance to learn standard model
    d_x2c = len(attr_names) # 108 concepts
    
    # known concept model: note here is a cbm model
    cbm = torch.load(f'{RootPath}/{concept_model_path}.pt')
    
    # unknown concept model
    if u_model_path:
        x2u = torch.load(f'{RootPath}/{u_model_path}.pt')
        x2u.aux_logits = False
    else:
        x2u = torch.hub.load('pytorch/vision:v0.9.0', 'inception_v3', pretrained=True)
        x2u.fc = nn.Linear(2048, d_x2u)
        x2u.aux_logits = False
        # x2u.AuxLogits.fc = nn.Linear(768, d_x2u)

    # combined model
    net = CCM_res(cbm, x2u)
    net.to(device)

    # print('task acc before training: {:.1f}%'.format(
    #     run_test(net, loader_xy_te) * 100))

    # print('task train acc before training: {:.1f}%'.format(
    #     run_test(net, loader_xy_eval) * 100))
    
    # add regularization to both u and c
    # lambda o, y, o_c, c, o_u:
    # F.cross_entropy(o, y) + 0.1 * (grad(o[y], o_u, create_graph=True)**2).sum()
    # + 0.1 * R_sq(c, o_u) # use c b/c o_c may not properly learned
    # make sure the 3 losses are at the same scale (is there a research question?)
    # and let dataset give x, y, c
    r = torch.cat([torch.ones(d_x2c), torch.zeros(d_x2u)]).to(device)
    criterion = lambda o_y, y: F.cross_entropy(o_y, y) # + \
    # * EYE(r, net_y[1].weight.abs().sum(0))
    
    # train
    opt = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=flags.wd)
    scheduler = lr_scheduler.StepLR(opt, step_size=lr_step)

    run_train = lambda **kwargs: train(
        net, loader_xy, opt, criterion=criterion,
        # shortcut specific
        shortcut_mode = flags.shortcut,
        shortcut_threshold = flags.threshold,
        n_shortcuts = flags.n_shortcuts,
        net_shortcut = net_s,
        # shortcut specific done
        independent=independent,
        n_epochs=n_epochs, report_every=report_every,
        device=device, savepath=savepath,
        scheduler=scheduler, **kwargs)

    if loader_xy_val:
        log  = run_train(
            report_dict={'val acc': (lambda m: run_test(m, loader_xy_val) * 100, 'max'),
                         'train acc': (lambda m: run_test(m, loader_xy_eval) * 100,
                                       'max')},
                    early_stop_metric='val acc')
    else:
        log = run_train(
            report_dict={'test acc': (lambda m: run_test(m, loader_xy_te) * 100, 'max'),
                         'train acc': (lambda m: run_test(m, loader_xy_eval) * 100,
                                       'max')})

    print('task acc after training: {:.1f}%'.format(run_test(net, loader_xy_te) * 100))
    return net

if __name__ == '__main__':
    flags = get_args()
    model_name = f"{RootPath}/{flags.outputs_dir}/ccm"
    print(model_name)

    # attributes to use
    attr_names = get_attr_names(f"{RootPath}/{flags.concept_path}")
    
    cub = CUB()
    test_indices = [i for i in range(len(cub)) if is_test_bird_idx(birdfile2idx(cub.images_path[i]))]    
    train_val_indices = [i for i in range(len(cub)) if not is_test_bird_idx(birdfile2idx(cub.images_path[i]))]
    train_val_labels = [cub.labels[i] for i in range(len(cub)) if not is_test_bird_idx(birdfile2idx(cub.images_path[i]))]
    val_ratio = 0.2
    train_indices, val_indices = train_test_split(train_val_indices, test_size=val_ratio,
                                                  stratify=train_val_labels,
                                                  random_state=flags.seed)
    
    # train indices already are used for training CBM, so we split validation in half to train the residual model
    if flags.split_val:
        train_indices, val_indices = train_test_split(val_indices, test_size=0.5,
                                                      random_state=flags.seed)

    # define dataloader: cub_train_eval is used to evaluate training data
    cub_train = CUB_train_transform(Subset(cub, train_indices), mode=flags.transform)
    cub_val = CUB_test_transform(Subset(cub, val_indices),  mode=flags.transform)
    cub_test = CUB_test_transform(Subset(cub, test_indices), mode=flags.transform)
    cub_train_eval = CUB_test_transform(Subset(cub, train_indices), mode=flags.transform)

    # accuracy
    acc_criterion = lambda o, y: (o.argmax(1) == y).float()

    # dataset, note only x, y not xyc
    subcolumn = lambda d: SubColumn(SubAttr(d, attr_names), ['x', 'y']) #, 'attr'])
    load = lambda d, shuffle: DataLoader(subcolumn(d), batch_size=32,
                                shuffle=shuffle, num_workers=8)
    loader_xy = load(cub_train, True)
    loader_xy_val = load(cub_val, False)
    loader_xy_te = load(cub_test, False)
    loader_xy_eval = load(cub_train_eval, False)
    
    print(f"# train: {len(cub_train)}, # val: {len(cub_val)}, # test: {len(cub_test)}")

    if flags.shortcut not in ['clean', 'noise']:
        net_s = torch.load(flags.shortcut)
    else:
        net_s = None
    
    run_train = lambda **kwargs:ccm(
        flags, attr_names, flags.c_model_path,
        loader_xy, loader_xy_eval,
        loader_xy_te, net_s=net_s,
        independent=flags.ind, alpha=flags.alpha,
        u_model_path = flags.u_model_path,
        n_epochs=flags.n_epochs, report_every=1,
        lr_step=flags.lr_step,
        savepath=model_name, use_aux=flags.use_aux, **kwargs)
    run_test = partial(test, 
                       criterion=acc_criterion, device='cuda',
                       # shortcut specific
                       shortcut_mode = flags.shortcut,
                       shortcut_threshold = flags.threshold,
                       n_shortcuts = flags.n_shortcuts,
                       net_shortcut = net_s)

    if flags.eval:
        print('task acc after training: {:.1f}%'.format(
            run_test(torch.load(f'{model_name}.pt'),
                     loader_xy_te) * 100))
    elif flags.retrain:
        cub_train = CUB_train_transform(Subset(cub, train_val_indices),
                                        mode=flags.transform)
        cub_train_eval = CUB_test_transform(Subset(cub, train_val_indices),
                                             mode=flags.transform)
        loader_xy = load(cub_train, True)
        loader_xy_eval = load(cub_train_eval, False)

        net = run_train()
    else:
        net = run_train(loader_xy_val=loader_xy_val)
        
