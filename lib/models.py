import torch.nn as nn
import types

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

