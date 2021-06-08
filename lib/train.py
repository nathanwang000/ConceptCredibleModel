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
          device="cpu", savepath=None, report_dict={}):
    '''
    report dict take the form: {"func_name": f} where f takes net as input
    an example is {"val_loss": lambda net: get_loss(net, val_loader)}
    '''
    net.train()
    train_log, losses = [], []

    for i in range(n_epochs):
        
        _losses = train_step(net, loader, opt, criterion, device=device)
        losses.extend(_losses)
        
        train_report = {"loss": np.mean(losses[-len(loader):])}
        train_report.update(dict((name, f(net)) for name, f in report_dict.items()))
        if (i+1) % report_every == 0: # report loss
            print('epoch {:>3}: '.format(i) + ' '.join('{} {:.3e}'.format(
                      name, val
                  ) for name, val in train_report.items()))

        train_report.update({'epoch': i})
        train_log.append(train_report)

        if savepath:
            torch.save(net, savepath + '.pt')
            torch.save(train_log, savepath + '.log')

    return train_log    

