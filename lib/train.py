import tqdm
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import torch
# custom import
from lib.models import CBM

def train_step_standard(net, loader, opt, criterion, device='cpu'):
    losses = []
    for x, y in tqdm.tqdm(loader, desc="train step for 1 epoch"):
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        o = net(x)
        try:
            o.shape
        except Exception as e:
            # for inception module
            o = o[0]
            
        l = criterion(o, y).mean()
        l.backward()
        opt.step()
        losses.append(l.detach().item())
    return losses
    
def train_step_xyz(net, loader, opt, criterion, device='cpu'):
    '''training step for EBM jont model'''
    assert type(net) == CBM, f"must use CBM model; currently {type(net)}"
    losses = []
    for x, y, z in loader:
        x, y, z = x.to(device), y.to(device), z.to(device)
        opt.zero_grad()
        o_z = net.net_c(x)
        o_y = net.net_y(o_z)
        l = criterion(o_y, y, o_z, z).mean()
        l.backward()
        opt.step()
        losses.append(l.detach().item())
    return losses

def train(net, loader, opt, train_step=train_step_standard, 
          criterion=F.cross_entropy, n_epochs=10, report_every=1,
          device="cpu", savepath=None, report_dict={},
          early_stop_metric=None, patience=10):
    '''
    report dict take the form: {"func_name": (f, mode)} where f takes net as input, mode is "max" or "min"
    an example is {"val_loss": lambda net: get_loss(net, val_loader)}
    
    es_metric: early stopping metric (string in report_dict)
    '''
    net.train()
    train_log, losses = [], []
    if early_stop_metric in report_dict:
        f, es_mode = report_dict[early_stop_metric]
        es = EarlyStopping(mode=es_mode, patience=patience)

    for i in range(n_epochs):
        
        _losses = train_step(net, loader, opt, criterion, device=device)
        losses.extend(_losses)
        
        train_report = {"loss": np.mean(losses[-len(loader):])}
        train_report.update(dict((name, f(net)) for name, \
                                 (f, es_mode) in report_dict.items()))
        if (i+1) % report_every == 0: # report loss
            print('epoch {:>3}: '.format(i) + ', '.join('{} {:.3e}'.format(
                      name, val
                  ) for name, val in train_report.items()))

        train_report.update({'epoch': i})
        train_log.append(train_report)

        if early_stop_metric in report_dict:
            metric = train_report[early_stop_metric]
            if es.step(metric):
                print("early stopping...")
                break
            
        if savepath:
            if early_stop_metric not in report_dict or es.num_bad_epochs==0:
                    torch.save(net, savepath + '.pt')
            torch.save(train_log, savepath + '.log')

    return train_log    

# MIT License for the following early stopping code
#
# Copyright (c) 2018 Stefano Nardo https://gist.github.com/stefanonardo
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

class EarlyStopping(object):
    
    def __init__(self, mode='min', min_delta=0, patience=10, percentage=False):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta, percentage)

        if patience == 0:
            self.is_better = lambda a, b: True
            self.step = lambda a: False

    def step(self, metrics):
        if self.best is None:
            self.best = metrics
            return False

        if np.isnan(metrics): # was torch.isnan(metrics)
            return True

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True

        return False

    def _init_is_better(self, mode, min_delta, percentage):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if not percentage:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - min_delta
            if mode == 'max':
                self.is_better = lambda a, best: a > best + min_delta
        else:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - (
                            best * min_delta / 100)
            if mode == 'max':
                self.is_better = lambda a, best: a > best + (
                            best * min_delta / 100)
