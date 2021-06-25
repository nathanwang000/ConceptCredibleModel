'''
this file trains a standard model
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

###
FilePath = os.path.dirname(os.path.abspath(__file__))
RootPath = os.path.dirname(FilePath)
if RootPath not in sys.path: # parent directory
    sys.path = [RootPath] + sys.path
from lib.models import MLP
from lib.data import small_CUB, CUB, SubColumn, CUB_train_transform, CUB_test_transform
from lib.train import train
from lib.eval import get_output, test, plot_log, shap_net_x, shap_ccm_c, bootstrap
from lib.utils import birdfile2class, birdfile2idx, is_test_bird_idx, get_bird_bbox, get_bird_class, get_bird_part, get_part_location, get_multi_part_location, get_bird_name
from lib.utils import get_attribute_name, code2certainty, get_class_attributes, get_image_attributes, describe_bird

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", action="store_true",
                        help="whether or not to eval the learned model")
    parser.add_argument("--retrain", action="store_true",
                        help="retrain using all train val data")
    parser.add_argument("--seed", type=int, default=42,
                        help="seed for reproducibility")
    parser.add_argument("--transform", default="crop",
                        help="transform mode to use")
    parser.add_argument("--lr_step", type=int, default=1000,
                        help="learning rate decay steps")
    parser.add_argument("--n_epochs", type=int, default=100,
                        help="max number of epochs to train")
    parser.add_argument("--use_aux", action="store_true",
                        help="auxilliary loss for inception")
    
    args = parser.parse_args()
    print(args)
    return args

def standard_model(loader_xy, loader_xy_eval, loader_xy_te, loader_xy_val=None,
                   n_epochs=10, report_every=1, lr_step=1000,
                   device='cuda', savepath=None, use_aux=False):
    '''
    loader_xy_eval is the evaluation of loader_xy
    if loader_xy_val: use early stopping, otherwise train for the number of epochs
    '''
    # regular model
    net = torch.hub.load('pytorch/vision:v0.9.0', 'inception_v3', pretrained=True)
    net.fc = nn.Linear(2048, 200) # 200 bird classes
    net.AuxLogits.fc = nn.Linear(768, 200)
    net.to(device)
    print('task acc before training: {:.1f}%'.format(test(net, loader_xy_te,
                                                          acc_criterion,
                                                          device=device) * 100))

    if use_aux:
        # for inception module where o[1] is auxilliary input to avoid vanishing
        # gradient https://stats.stackexchange.com/questions/274286/google-inception-modelwhy-there-is-multiple-softmax
        # https://discuss.pytorch.org/t/why-auxiliary-logits-set-to-false-in-train-mode/40705
        criterion = lambda o, y: F.cross_entropy(o[0], y) + \
            0.4 * F.cross_entropy(o[1], y)
    else:
        criterion = lambda o, y: F.cross_entropy(o[0], y)
    
    # train
    opt = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0004)
    scheduler = lr_scheduler.StepLR(opt, step_size=lr_step)
    if loader_xy_val:
        log = train(net, loader_xy, opt, criterion=criterion,
                    n_epochs=n_epochs, report_every=report_every,
                    device=device, savepath=savepath,
                    report_dict={'val acc': (lambda m: test(m, loader_xy_val,
                                                            acc_criterion,
                                                            device=device) * 100, 'max'),
                                 'train acc': (lambda m: test(m, loader_xy_eval,
                                                              acc_criterion,
                                                              device=device) * 100,
                                               'max')},
                    early_stop_metric='val acc',
                    scheduler=scheduler)
    else:
        log = train(net, loader_xy, opt, criterion=criterion,
                    n_epochs=n_epochs, report_every=report_every,
                    device=device, savepath=savepath,
                    report_dict={'train acc': (lambda m: test(m, loader_xy_eval,
                                                            acc_criterion,
                                                            device=device) * 100, 'max'),
                                 'test acc': (lambda m: test(m, loader_xy_te,
                                                              acc_criterion,
                                                              device=device) * 100,
                                               'max')},
                    scheduler=scheduler)

    print('task acc after training: {:.1f}%'.format(test(net, loader_xy_te,
                                                         acc_criterion,
                                                         device=device) * 100))        
    return net

if __name__ == '__main__':
    flags = get_args()
    model_name = f"{RootPath}/models/standard"
    print(model_name)

    cub = CUB()
    test_indices = [i for i in range(len(cub)) if is_test_bird_idx(birdfile2idx(cub.images_path[i]))]    
    train_val_indices = [i for i in range(len(cub)) if not is_test_bird_idx(birdfile2idx(cub.images_path[i]))]
    train_val_labels = [cub.labels[i] for i in range(len(cub)) if not is_test_bird_idx(birdfile2idx(cub.images_path[i]))]
    val_ratio = 0.2
    train_indices, val_indices = train_test_split(train_val_indices, test_size=val_ratio,
                                                  stratify=train_val_labels,
                                                  random_state=flags.seed)

    # define dataloader: cub_train_eval is used to evaluate training data
    cub_train = CUB_train_transform(Subset(cub, train_indices), mode=flags.transform)
    cub_val = CUB_test_transform(Subset(cub, val_indices),  mode=flags.transform)
    cub_test = CUB_test_transform(Subset(cub, test_indices), mode=flags.transform)
    cub_train_eval = CUB_test_transform(Subset(cub, train_indices), mode=flags.transform)

    # accuracy
    acc_criterion = lambda o, y: (o.argmax(1) == y).float()

    # dataset
    loader_xy = DataLoader(SubColumn(cub_train, ['x', 'y']), batch_size=32,
                           shuffle=True, num_workers=8)
    loader_xy_val = DataLoader(SubColumn(cub_val, ['x', 'y']), batch_size=32,
                               shuffle=False, num_workers=8)
    loader_xy_te = DataLoader(SubColumn(cub_test, ['x', 'y']), batch_size=32,
                              shuffle=False, num_workers=8)
    loader_xy_eval = DataLoader(SubColumn(cub_train_eval, ['x', 'y']), batch_size=32,
                                shuffle=True, num_workers=8)

    print(f"# train: {len(cub_train)}, # val: {len(cub_val)}, # test: {len(cub_test)}")

    if flags.eval:
        print('task acc after training: {:.1f}%'.format(
            test(torch.load(f'{model_name}.pt'),
                 loader_xy_te, acc_criterion, device='cuda') * 100))
    elif flags.retrain:
        cub_train = CUB_train_transform(Subset(cub, train_val_indices),
                                        mode=flags.transform)
        cub_train_eval = CUB_test_transform(Subset(cub, train_val_indices),
                                        mode=flags.transform)
        loader_xy = DataLoader(SubColumn(cub_train, ['x', 'y']), batch_size=32,
                               shuffle=True, num_workers=8)
        loader_xy_eval = DataLoader(SubColumn(cub_train_eval, ['x', 'y']), batch_size=32,
                                    shuffle=True, num_workers=8)
        standard_net = standard_model(loader_xy, loader_xy_eval,
                                      loader_xy_te,
                                      n_epochs=flags.n_epochs, report_every=1,
                                      lr_step=flags.lr_step,
                                      savepath=model_name, use_aux=flags.use_aux)
    else:
        standard_net = standard_model(loader_xy, loader_xy_eval,
                                      loader_xy_te, loader_xy_val=loader_xy_val,
                                      n_epochs=flags.n_epochs, report_every=1,
                                      lr_step=flags.lr_step,
                                      savepath=model_name, use_aux=flags.use_aux)
        
