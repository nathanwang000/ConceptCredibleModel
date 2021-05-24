import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
# custom import
from lib.models import CBM

def train_step_standard(net, loader, opt, criterion, device='cpu'):
    losses = []
    for x, y in loader:
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
          device="cpu"):
    net.train()
    train_log, losses = [], []

    for i in range(n_epochs):
        
        _losses = train_step(net, loader, opt, criterion, device=device)
        losses.extend(_losses)
        
        train_report = {"loss": np.mean(losses[-len(loader):])}
        if (i+1) % report_every == 0: # report loss
            print('epoch {:>3}: '.format(i) + ' '.join('{} {:.3e}'.format(
                      name, val
                  ) for name, val in train_report.items()))

        train_report.update({'epoch': i})
        train_log.append(train_report)

    return train_log    

