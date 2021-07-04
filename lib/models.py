import torch
import torch.nn as nn
import types

####
from lib.utils import attribute2idx

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
    def __init__(self, net_c, net_u, net_y, c_no_grad=True, u_no_grad=False):
        '''
        f(x) = net_y(net_c(x), net_u(x))
        net_c and net_not_c output (n, d_*)
        c_no_grad: known concept model doesn't need gradient (save gpu memory)
        u_no_grad: unknown concept model doesn't need gradient
        '''
        super().__init__()
        self.net_c = net_c
        self.net_u = net_u
        self.net_y = net_y
        self.c_no_grad = c_no_grad
        self.u_no_grad = u_no_grad

    def get_oc(self, x):

        if self.c_no_grad:
            with torch.no_grad():
                self.net_c.eval()
                o_c = self.net_c(x)
        else:
            o_c = self.net_c(x)
            
        return o_c

    def get_ou(self, x):

        if self.u_no_grad:
            with torch.no_grad():
                self.net_u.eval()
                o_u = self.net_u(x)
        else:
            o_u = self.net_u(x)
        return o_u

    def forward(self, x):
        o_c = self.get_oc(x)
        o_u = self.get_ou(x)
        o_y = self.net_y([o_c, o_u])
        return o_y
    
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
        

