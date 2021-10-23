import torch
import torch.nn as nn
import types
import numpy as np
from torchvision import transforms

####
from lib.utils import attribute2idx, dfs_freeze, get_shortcut_level

class CDropout(nn.Module):
    '''
    dropout layer with C
    r: binary risk factor array
    p: dropout rate
    '''
    def __init__(self, r, p):
        assert 0 <= p <=1, f"p need to be within [0, 1], now is {p}"
        super().__init__()
        np.random.seed(42)
        self.r = r
        self.p = p
        mask = torch.from_numpy(np.random.choice(2,len(r),
                                                 p=[1-p, p])).to(r.device)
        # mask represent features to throw away
        self.mask = (mask * (1-r)).bool() # keep risk factors

    def forward(self, x):
        # import pdb; pdb.set_trace()        
        return x * (~self.mask) # broadcast to last dimension
    
class MLP(nn.Module):

    def __init__(self, neuron_sizes, activation=nn.ReLU, bias=True): 
        super().__init__()
        self.neuron_sizes = neuron_sizes
        
        layers = []
        for s0, s1 in zip(neuron_sizes[:-1], neuron_sizes[1:]):
            layers.extend([
                nn.Linear(s0, s1, bias=bias),
                activation()
            ])
        
        self.classifier = nn.Sequential(*layers[:-1])
        
    def forward(self, x):
        return self.classifier(x)

class MTL(nn.Module):
    '''
    custom multi-task-learng model
    c_net: is the c model
    net_base: base for network, shared for predict C and Y
    net_c: use output of net_base to predict C
    net_y: use output of net_base to predict Y

    training mode: output y_hat, c_hat, c
    test_mode: output y_hat
    '''
    def __init__(self, c_net, net_base, net_c, net_y):
        super().__init__()
        dfs_freeze(c_net) # don't update its parameter
        self.c_net = c_net
        self.net_base = net_base
        self.net_c = net_c
        self.net_y = net_y

    def forward(self, x):
        if self.training: # training mode
            with torch.no_grad():
                self.c_net.eval()
                c = self.c_net(x)
            o = self.net_base(x)
            return self.net_y(o), self.net_c(o), c
        return self.net_y(self.net_base(x))
    
class ENS(nn.Module):
    
    def __init__(self, models):
        '''
        ensemble model
        '''
        super().__init__()
        self.models = models
        
    def forward(self, x):
        return sum([net(x) for net in self.models]) / len(self.models)
    
class LambdaNet(nn.Module):
    def __init__(self, lambd):
        super().__init__()
        import types
        assert type(lambd) is types.LambdaType
        self.lambd = lambd

    def forward(self, *args):
        return self.lambd(*args)

class ConcatNet(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, args):
        return torch.cat(args, dim=self.dim)

class SubColumnNet(nn.Module):
    def __init__(self, columns):
        '''
        get subset of input
        '''
        super().__init__()
        self.columns = columns

    def forward(self, x):
        return x[:, self.columns]

######### specific models
class CUB_Subset_Concept_Model(nn.Module):
    '''
    learned concept transition into attr_names
    '''
    def __init__(self, attr_subset_names, attr_full_names):
        super().__init__()
        self.attr_idx = [attr_full_names.index(name) for name in attr_subset_names]

    def forward(self, x):
        return x[:, self.attr_idx]

class CUB_Noise_Concept_Model(nn.Module):
    '''
    replaces first d concepts with noise N(0, 1)
    '''
    def __init__(self, d):
        super().__init__()
        self.d = d

    def forward(self, x):
        bs, d = x.shape
        d = min(d, self.d)
        noise = torch.randn(bs, d).to(x.device)
        return torch.cat((noise, x[:, d:]), 1)

class Concat_CS_Model(nn.Module):
    '''
    in addition to C, concatenate S to the model
    
    net_s is the shortcut model
    net is the main branch of the model
    threshold: threshold as in utils:get_shortcut_level
    '''
    def __init__(self, net, net_s, n_shortcuts, threshold=1.0):
        super().__init__()
        self.net_s = net_s
        self.net = net
        self.n_shortcuts = n_shortcuts
        self.threshold = threshold

    def forward(self, x):
        o = self.net(x)
        with torch.no_grad():
            self.net_s.eval()
            # (bs,)
            shortcut_level = get_shortcut_level(self.net_s(x).argmax(1),
                                                self.threshold,
                                                self.n_shortcuts)
            s = torch.nn.functional.one_hot(shortcut_level, num_classes=self.n_shortcuts)
        return torch.cat((s, o), 1)
    
class GT_CUB_Subset_Concept_Model(nn.Module):
    '''
    ground truth concept model but subset of concepts in CUB
    '''
    def __init__(self, attributes_names):
        super().__init__()
        self.net = MLP([len(attributes_names), 200]) # logistic regression
        # doesn't matter much b/c class attribute is fixed
        # self.net = MLP([len(attributes_names), 30, 30, 200])
        self.attr_idx = [attribute2idx(a)-1 for a in attributes_names] # 0-index

    def forward(self, x):
        # assume x is attributes
        x = x[:, self.attr_idx]
        return self.net(x)

class CCM(nn.Module):
    '''
    concept credible model
    '''
    def __init__(self, net_c, net_u, net_y, c_no_grad=True,
                 u_no_grad=False, mask=False):
        '''
        f(x) = net_y(net_c(x), net_u(x))
        net_c and net_not_c output (n, d_*)
        c_no_grad: known concept model doesn't need gradient (save gpu memory)
        u_no_grad: unknown concept model doesn't need gradient
        mask: unknown concept has different data augmentation
        '''
        super().__init__()
        self.net_c = net_c
        self.net_u = net_u
        self.net_y = net_y
        self.c_no_grad = c_no_grad
        self.u_no_grad = u_no_grad
        self.mask = mask
        if c_no_grad:
            dfs_freeze(self.net_c)
        if u_no_grad:
            dfs_freeze(self.net_u)            

    def get_oc(self, x):

        if self.c_no_grad:
            with torch.no_grad():
                self.net_c.eval()
                o_c = self.net_c(x)
                o_c.requires_grad = True # get ig methods working
        else:
            o_c = self.net_c(x)
            
        return o_c

    def get_ou(self, x):

        if hasattr(self, 'mask') and self.mask and self.training:
            # only do this in training
            x = transforms.RandomErasing()(x)
        
        if self.u_no_grad:
            with torch.no_grad():
                self.net_u.eval()
                o_u = self.net_u(x)
                o_u.requires_grad = True # get ig methods working
        else:
            o_u = self.net_u(x)

        return o_u

    def forward(self, x):
        o_c = self.get_oc(x)
        o_u = self.get_ou(x)
        o_y = self.net_y([o_c, o_u])
        return o_y

class CCM_res(nn.Module):
    '''
    residual CCM
    '''
    def __init__(self, net1, net2): 
        super().__init__()
        self.net1 = net1
        dfs_freeze(self.net1)
        self.net2 = net2

    def forward(self, x):
        with torch.no_grad():
            self.net1.eval()
            o = self.net1(x)
        return o + self.net2(x)
    
class CBM(nn.Module):
    '''
    concept bottleneck model
    net_c is concept net, net_y is the task net
    it output net_y(net_c(x))
    '''

    def __init__(self, net_c, net_y, c_no_grad=True): 
        super().__init__()
        self.net_c = net_c
        self.net_y = net_y
        self.c_no_grad = c_no_grad
        if c_no_grad:
            dfs_freeze(self.net_c)

    def get_oc(self, x):
        if self.c_no_grad:
            with torch.no_grad():
                self.net_c.eval()
                o_c = self.net_c(x)
        else:
            o_c = self.net_c(x)
        return o_c

    def forward(self, x):
        o_c = self.get_oc(x)
        o_y = self.net_y(o_c)
        return o_y
        

