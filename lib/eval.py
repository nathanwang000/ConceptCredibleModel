import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
# custom import
from lib.models import CCM

def get_output(net, loader_x):
    o = []
    for x, in loader_x:
        o.extend(net(x).detach().numpy())
    return np.vstack(o) # (n, c)
                        
def test(net, loader, criterion):
    net.eval()
    losses = []
    total = 0
    for x, y in loader:
        o = net(x)
        l = criterion(o, y).mean()
        bs = o.shape[0]
        total += bs        
        losses.append(l.detach().item() * bs)
    net.train()
    return sum(losses) / total

def plot_log(log, key="loss"):
    '''log is train log [{epoch: xxx, loss: xxx}]'''
    x = [item['epoch'] for item in log]
    y = [item[key] for item in log]
    plt.semilogy(x, y)
    plt.xlabel('epochs', fontsize=15)
    plt.ylabel(key, fontsize=15)
    plt.plot()    

def shap_net_x(net, shap_x, bs, instance_idx=None, output_idx=None, decimal=2, output_name="hat Y"):
    '''
    explain net on input level
    '''
    prediction = get_output(net, 
                            DataLoader(TensorDataset(torch.from_numpy(np.array(shap_x))), 
                                       batch_size=bs, shuffle=False))
    d_o = prediction.shape[1]
    df_o = pd.DataFrame(dict((f"{output_name}{i}", np.around(prediction[:, i], decimal)) for i in range(d_o)))
    df_o.set_index(shap_x.index, inplace=True)

    print(
        pd.concat((shap_x, df_o), 
                  1).sort_values(f"{output_name}0"))

    def explain_one_output(i):
        print(f'Explaining {output_name}{i}')
        
        # shap explanation
        explainer = shap.KernelExplainer(lambda x: net(torch.from_numpy(x)).detach()[:, i].numpy(), 
                                         shap_x)
        shap_values = explainer.shap_values(shap_x, nsamples=100)
        if instance_idx != None:
            shap.summary_plot(shap_values[instance_idx:instance_idx+1], shap_x[instance_idx:instance_idx+1])
        else:
            shap.summary_plot(shap_values, shap_x)

    if output_idx:
        explain_one_output(output_idx)
    else:
        for i in range(d_o):
            explain_one_output(i)

def shap_ccm_c(ccm, shap_x, bs, instance_idx=None, output_idx=None, c_name='Z', not_c_name='U', output_name='hat Y', decimal=2):
    '''
    explain ccm at concept level: the first part is interpretable (c_name), the second part is not (not_c_name)
    shap_x is pd.DataFrame to explain using shap
    '''
    assert type(ccm) is CCM, "this function only applies to CCM model"
    explain_X = torch.from_numpy(np.array(shap_x))
    
    # compute the concepts: z interpretable, u not interpretable
    z_hat = get_output(ccm.net_c,
                       DataLoader(TensorDataset(explain_X), batch_size=bs, shuffle=False))
    c = z_hat.shape[1]
    z_hat = pd.DataFrame(z_hat, columns=[f'{c_name}{i}' for i in range(c)])
    u_hat = get_output(ccm.net_not_c,
                       DataLoader(TensorDataset(explain_X), batch_size=bs, shuffle=False))
    not_c = u_hat.shape[1]
    u_hat = pd.DataFrame(u_hat, columns=[f'{not_c_name}{i}' for i in range(not_c)])
    shap_zu = pd.concat((z_hat, u_hat), 1)
                
    # print prediction
    prediction = get_output(nn.Sequential(ccm, nn.Softmax(dim=1)),
                            DataLoader(TensorDataset(explain_X), batch_size=bs, shuffle=False))
    d_o = prediction.shape[1]
    print(pd.concat((shap_zu.round(decimal),
                     pd.DataFrame(dict((f"{output_name}{i}", 
                                        np.around(prediction[:, i], decimal)) for i in range(d_o)))), 
                    1).sort_values(f'{output_name}0'))

    def explain_one_output(i):
        print(f'Explaining {output_name}{i}')
    
        ccm_concept = lambda x: F.softmax(ccm.net_y(
            torch.from_numpy(x[:, :c]),
            torch.from_numpy(x[:, -not_c:])
        ), 1).detach()[:, i].numpy()

        # shap explanation
        explainer = shap.KernelExplainer(ccm_concept, 
                                         shap_zu)
        shap_values = explainer.shap_values(shap_zu, nsamples=100)

        if instance_idx:
            shap.summary_plot(shap_values[instance_idx:instance_idx+1], shap_zu[instance_idx:instance_idx+1])
        else:
            shap.summary_plot(shap_values, shap_zu)
        
    if output_idx:
        explain_one_output(output_idx)
    else:
        for i in range(d_o):
            explain_one_output(i)
