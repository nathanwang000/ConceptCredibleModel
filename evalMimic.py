# mimic chxpert evaluation
'''
given a model, test it performance on the biased dataset and
the clean dataset;

The bias dataset are created by changing the model's last layer
representation (suppressing the output of neurons that are highly
correlated with C)
'''
import os, sys
import torch
from torch import nn
import numpy as np
import argparse
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from torch.utils.data import TensorDataset, DataLoader
from functools import partial

###
FilePath = os.path.dirname(os.path.abspath(__file__))
RootPath = os.path.dirname(FilePath)
if RootPath not in sys.path: # parent directory
    sys.path = [RootPath] + sys.path
from lib.data import SubColumn, MIMIC_train_transform, MIMIC_test_transform
from lib.data import MIMIC
from lib.eval import get_output, test_auc
from lib.models import CDropout, CCM, ConcatNet, CCM_res, CBM

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("net_path",
                        help="path to the network")
    parser.add_argument("-p", default=0.5, type=float,
                        help="threshold used for custom dropout")
    parser.add_argument("--task", default='Edema',
                        help="which task of mimic to run")
    parser.add_argument("--seed", type=int, default=42,
                        help="seed for reproducibility")
    parser.add_argument("--transform", default="cbm",
                        help="transform mode to use")
    
    args = parser.parse_args()
    print(args)
    return args

def get_mimic_data(flags):
    task = flags.task
    mimic = MIMIC(task) # mimic doesn't have validation data, chexpert has
    indices = list(range(len(mimic)))
    labels = list(mimic.df[task])

    test_ratio = 0.2
    train_val_indices, test_indices, train_val_labels, _ = train_test_split(
        indices, labels, test_size=test_ratio, stratify=labels,
        random_state=flags.seed)
    val_ratio = 0.2

    train_indices, val_indices = train_test_split(train_val_indices,
                                                  test_size=val_ratio,
                                                  stratify=train_val_labels,
                                                  random_state=flags.seed)
    

    # define dataloader: mimic_train_eval is used to evaluate training data
    mimic_train = MIMIC_train_transform(Subset(mimic, train_indices),
                                        mode=flags.transform)
    mimic_val = MIMIC_test_transform(Subset(mimic, val_indices),
                                     mode=flags.transform)
    mimic_test = MIMIC_test_transform(Subset(mimic, test_indices),
                                      mode=flags.transform)
    mimic_train_eval = MIMIC_test_transform(Subset(mimic, train_indices),
                                            mode=flags.transform)

    # dataset    
    subcolumn = lambda d: SubColumn(d, ['x', 'y'])
    load = lambda d, shuffle: DataLoader(subcolumn(d), batch_size=32,
                                         shuffle=shuffle, num_workers=8,
                                         drop_last=True)
    loader_xy = load(mimic_train, True)
    loader_xy_val = load(mimic_val, False)
    loader_xy_te = load(mimic_test, False)
    loader_xy_eval = load(mimic_train_eval, False)
    
    print(f"# train: {len(mimic_train)}, # val: {len(mimic_val)}, # test: {len(mimic_test)}")
    return loader_xy_eval, loader_xy_val, loader_xy_te
    
if __name__ == '__main__':
    flags = get_args()
    tr_loalder, val_loader, te_loader = get_mimic_data(flags)

    # get the model
    net = torch.load(flags.net_path)

    # sabotage the fc layer repr by correlation with C
    # this works for std(c,x), std(x), ccm_eye when dimensions are
    # sabotaged based on correlation
    # but also works for ccm_res if dimensions are sabotaged by random
    # dimensions (dropout with different dropout rate) on non-c features
    
    # add a custom dropout layer that only in train mode
    # and then average over 100 random dropout (run at most 90 min)
    # this make sense because we cannot pin point the shortcut in the
    # image to really test out its usefulness; however, we can always
    # mask out concepts in the input images; to avoid it affecting, C,
    # we do the dropout at the last layer

    device = 'cuda'
    run_test = partial(test_auc, 
                       device=device,
                       max_batches= 100, #None if flags.eval else 100,
                       # shortcut specific
                       shortcut_mode = 'clean')
    
    # biased dataset
    print('biased task auc: {:.1f}%'.format(
        run_test(net,
                 te_loader) * 100))

    
    # clean dataset
    if type(net) is CCM: # std(c,x), eye
        r = torch.zeros(2048 * 2).to(device)
        r[:2048] = 1
        net.net_y = nn.Sequential(net.net_y[0], # concat net
                                  CDropout(r, flags.p),
                                  net.net_y[1])
    elif type(net) is CCM_res: # res
        r = torch.zeros(2048).to(device)
        net.net2.fc = nn.Sequential(CDropout(r, flags.p), net.net2.fc)
    elif type(net) is CBM: # cbm
        print("CBM clean and bias result is the same")
    else: # std(x)
        print('STD(X)')
        r = torch.zeros(2048).to(device)
        net.fc = nn.Sequential(CDropout(r, flags.p), net.fc)
    
    print('clean task auc: {:.1f}%'.format(
        run_test(net,
                 te_loader) * 100))
