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

######### specific models
class GT_CUB_Subset_Concept_Model(nn.Module):
    '''
    ground truth concept model but subset of concepts in CUB
    '''
    def __init__(self, attributes_names):
        super().__init__()
        self.net = MLP([len(attributes_names), 200]) # logistic regression
        self.attr_idx = [attribute2idx(a)-1 for a in attributes_names] # 0-index

    def forward(self, x):
        # assume x is attributes
        x = x[:, self.attr_idx]
        return self.net(x)
        
class CCM(nn.Module):
    '''
    concept credible model
    '''
    def __init__(self, net_c, net_not_c, net_y):
        '''
        f(x) = net_y(net_c(x), net_not_c(x))
        net_c and net_not_c output (n, d_*)
        '''
        super().__init__()
        self.net_c = net_c
        self.net_not_c = net_not_c
        self.net_y = net_y
        
    def forward(self, x):
        return self.net_y(self.net_c(x), self.net_not_c(x))
    
class CBM(nn.Module):
    '''
    concept bottleneck model
    net_c is concept net, net_y is the task net
    it output net_y(net_c(x))
    '''

    def __init__(self, net_c, net_y): 
        super().__init__()
        self.net_c = net_c
        self.net_y = net_y
        
        self.classifier = nn.Sequential(net_c, net_y)
        
    def forward(self, x):
        return self.classifier(x)

