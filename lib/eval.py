import shap
import math
import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn import metrics, utils
from joblib import Parallel, delayed

# custom import
from lib.models import CCM
from lib.utils import attribute2idx
from lib.data import CUB_shortcut_transform

@torch.no_grad()
def get_output(net, loader):
    net.eval()
    try:
        device = net.parameters().__next__().device
    except:
        device = 'cpu'
    
    o = []
    for d in loader:
        if type(d) is dict: x = d['x']
        else: x = d[0]
        x = x.to(device)
        o.extend(net(x).cpu().detach().numpy())
    return np.vstack(o) # (n, c)

def bootstrap(metric, y, y_hat, l=2.5, h=97.5, n=100, n_jobs=4):
    '''
    https://ocw.mit.edu/courses/mathematics/18-05-introduction-to-probability-and-statistics-spring-2014/readings/MIT18_05S14_Reading24.pdf
    metric: (y, y_hat) -> R+
    l and high are percentiles
    n_jobs: number of parallel jobs
    '''
    def b_helper(y, y_hat):
        return metric(*utils.resample(y, y_hat, replace=True))
    
    s = metric(y, y_hat)    
    scores = Parallel(n_jobs=n_jobs)(
        delayed(b_helper)(y, y_hat) for i in range(n))
    scores = np.array(scores)
    delta = scores - s
    dl, dh = np.percentile(delta, (l, h))
    return s - dh, s - dl

@torch.no_grad()
def test(net, loader, criterion, device='cpu', **kwargs):
    net.eval()
    losses = []
    total = 0
    for d in tqdm.tqdm(loader, desc="test eval"):
        x, y = d[0], d[1] # assume the first 2 are x, y to accomendate xyc
        x, y = x.to(device), y.to(device)
        x, s = CUB_shortcut_transform(x, y, **kwargs)        
        o = net(x)
        l = criterion(o, y).mean()
        bs = o.shape[0]
        total += bs        
        losses.append(l.detach().item() * bs)
    net.train()
    return sum(losses) / total

@torch.no_grad()
def test_auc(net, loader, device='cpu', **kwargs):
    net.eval()
    outputs, truths = [], []
    n_batches = 0
    for d in tqdm.tqdm(loader, desc="test eval"):
        n_batches += 1
        if kwargs.get("max_batches", None) and n_batches > kwargs["max_batches"]:
            break
        
        x, y = d[0], d[1] # assume the first 2 are x, y to accomendate xyc
        x, y = x.to(device), y.to(device)
        x, s = CUB_shortcut_transform(x, y, **kwargs)
        o = net(x)
        if o.shape[1] == 1:
            o = torch.sigmoid(o)[:, 0]
        else:
            o = F.softmax(net(x), 1)[:, 1] # take the positive probability
        outputs.append(o.detach().cpu().numpy())
        truths.append(y.detach().cpu().numpy())
    net.train()
    return metrics.roc_auc_score(np.hstack(truths), np.hstack(outputs))

def plot_log(log, key="loss", semi_y=False, label=None):
    '''log is train log [{epoch: xxx, loss: xxx}]'''
    x = [item['epoch'] for item in log]
    y = [item[key] for item in log]
    if semi_y:
        plt.semilogy(x, y, label=label)
    else:
        plt.plot(x, y, label=label)
    plt.xlabel('epochs', fontsize=15)
    plt.ylabel(key, fontsize=15)


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

def shap_ccm_c(ccm, shap_x, bs, instance_idx=None, output_idx=None, c_name='Z', u_name='U', output_name='hat Y', decimal=2):
    '''
    explain ccm at concept level: the first part is interpretable (c_name), the second part is not (u_name)
    shap_x is pd.DataFrame to explain using shap
    '''
    assert type(ccm) is CCM, "this function only applies to CCM model"
    explain_X = torch.from_numpy(np.array(shap_x))
    
    # compute the concepts: z interpretable, u not interpretable
    z_hat = get_output(ccm.net_c,
                       DataLoader(TensorDataset(explain_X), batch_size=bs, shuffle=False))
    c = z_hat.shape[1]
    z_hat = pd.DataFrame(z_hat, columns=[f'{c_name}{i}' for i in range(c)])
    u_hat = get_output(ccm.net_u,
                       DataLoader(TensorDataset(explain_X), batch_size=bs, shuffle=False))
    u = u_hat.shape[1]
    u_hat = pd.DataFrame(u_hat, columns=[f'{u_name}{i}' for i in range(u)])
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
            [torch.from_numpy(x[:, :c]),
            torch.from_numpy(x[:, -u:])]
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


# other attribution methods adapted from https://github.com/utkuozbulak/pytorch-cnn-visualizations/blob/master/src/vanilla_backprop.py; note that the github implementation is incorrect (e.g., integrated gradient didn't scale by image and assumes model has features attributes); mine is better
class VanillaBackprop():
    """
        Produces gradients generated with vanilla back propagation from the image
        adapted from https://github.com/utkuozbulak/pytorch-cnn-visualizations/blob/master/src/vanilla_backprop.py
    """
    def __init__(self, model):
        self.model = model
        # Put model in evaluation mode
        self.model.eval()

    def explain(self, input_image, target_class):
        # generate_gradients
        # Forward
        input_image.requires_grad = True # must so that input can have grad
        model_output = self.model(input_image)
        # Zero grads
        self.model.zero_grad()
        # Target for backprop
        one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_()
        one_hot_output[0][target_class] = 1
        # Backward pass
        model_output.backward(gradient=one_hot_output.to(model_output.device))
        # Convert Pytorch variable
        # [0] to get rid of the first channel (1,3,224,224)
        gradients_as_arr = input_image.grad.data[0].cpu() # self.gradients.data.numpy()[0]
        return gradients_as_arr

class IntegratedGradients():
    """
        Produces gradients generated with integrated gradients from the image
    """
    def __init__(self, model, steps):
        self.model = model
        self.steps = steps
        # Put model in evaluation mode
        self.model.eval()

    def generate_images_on_linear_path(self, input_image, steps):
        # Generate uniform numbers between 0 and steps
        step_list = np.arange(steps+1)/steps
        # Generate scaled xbar images: assumes 0 is the background image
        xbar_list = [input_image*step for step in step_list]
        return xbar_list

    def generate_gradients(self, input_image, target_class):
        # Forward
        input_image.requires_grad = True # must so that input can have grad
        model_output = self.model(input_image)
        # Zero grads
        self.model.zero_grad()
        # Target for backprop
        one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_()
        one_hot_output[0][target_class] = 1
        # Backward pass
        model_output.backward(gradient=one_hot_output.to(model_output.device))
        # Convert Pytorch variable
        # [0] to get rid of the first channel (1,3,224,224)
        gradients_as_arr = input_image.grad.data[0].cpu() # self.gradients.data.numpy()[0]
        return gradients_as_arr
        
    def explain(self, input_image, target_class):
        # generate_integrated_gradients
        steps = self.steps
        # Generate xbar images
        xbar_list = self.generate_images_on_linear_path(input_image, steps)
        # Initialize an image composed of zeros
        integrated_grads = torch.zeros_like(input_image).cpu()
        for xbar_image in xbar_list:
            # Generate gradients from xbar images
            single_integrated_grad = self.generate_gradients(xbar_image, target_class)
            # Add rescaled grads from xbar images
            integrated_grads = integrated_grads + single_integrated_grad/steps
        # [0] to get rid of the first channel (1,3,224,224)
        # return integrated_grads[0] # should probably times input image by the original paper
        # this differs from the online implementation (their version is wrong b/c it should be scaled by size along each direction, not a global step)
        return (integrated_grads * input_image.cpu())[0] # background value of 0

def show_attribution(dataset, models, attributes, idx, explain_method=VanillaBackprop, n_col=5, device='cuda'):
    '''
    show feature attribution of models to input
    '''
    n_row = math.ceil(float(len(attributes)+1) / n_col)
    im, y, attr = dataset[idx]['x'].permute(1,2,0), dataset[idx]['y'], dataset[idx]['attr']
    print('0-indexed class id (describe bird is 1-indexed):', y)
    plt.figure(figsize=(4 * n_col, n_row * 3))
    plt.subplot(n_row, n_col, 1)
    plt.imshow((im - im.min()) / (im.max() - im.min()))
    plt.axis('off')

    for i, m in enumerate(models):
        explain = explain_method(m) # IntegratedGradients(m)
        # Generate attribution
        target_class = int(attr[attribute2idx(attributes[i]) - 1].item())
        attribution = explain.explain(dataset[idx]['x'].unsqueeze(0).to(device),
                                      target_class)

        plt.subplot(n_row, n_col, i+2)
        # save as convert2grayscale image in the online visualization code
        grad = attribution.permute(1,2,0).abs().sum(-1)
        plt.imshow((grad - grad.min()) / (grad.max() - grad.min()), cmap='twilight')
        plt.title(f"{attributes[i]}: {target_class}", )
        plt.axis('off')
    plt.show()

def show_explanation(dataset, idx, models, explain_method=VanillaBackprop, n_col=5, device='cuda'):
    '''
    show feature attribution of models to input
    '''
    n_row = math.ceil(float(len(models)+1) / n_col)
    im, y, attr = dataset[idx]['x'].permute(1,2,0), dataset[idx]['y'], dataset[idx]['attr']
    target_class = y
    print('class id:', y)
    plt.figure(figsize=(4 * n_col, n_row * 3))
    plt.subplot(n_row, n_col, 1)
    plt.imshow((im - im.min()) / (im.max() - im.min()))
    plt.axis('off')

    for i, m in enumerate(models):
        explain = explain_method(m) # IntegratedGradients(m)
        # Generate attribution
        x = dataset[idx]['x'].unsqueeze(0).to(device)
        pred = m(x).argmax(1).item()
        attribution = explain.explain(x,
                                      target_class)

        plt.subplot(n_row, n_col, i+2)
        # save as convert2grayscale image in the online visualization code
        grad = attribution.permute(1,2,0).abs().sum(-1)
        print(f'l2^2(grad): {(grad**2).sum()}')
        # plt.imshow(grad / grad.max(), cmap='twilight')
        plt.imshow((grad - grad.min()) / (grad.max() - grad.min()), cmap='twilight')
        
        plt.title(f"{target_class}: pred {pred}" )
        plt.axis('off')
    plt.show()
    

