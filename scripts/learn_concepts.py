import sys, os
import tqdm
import numpy as np
import seaborn as sns
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from torchvision import transforms
from IPython.display import Image as showImg
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Subset
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from copy import deepcopy
import matplotlib
import pandas as pd
import math
from sklearn.model_selection import train_test_split

###
FilePath = os.path.dirname(os.path.abspath(__file__))
RootPath = os.path.dirname(FilePath)
if RootPath not in sys.path: # parent directory
    sys.path = [RootPath] + sys.path
from lib.data import small_CUB, CUB, SubColumn, CUB_train_transform, CUB_test_transform, SubAttr
from lib.models import MLP
from lib.data import small_CUB, CUB, SubColumn
from lib.train import train
from lib.eval import get_output, test, plot_log, shap_net_x, shap_ccm_c, bootstrap, show_attribution
from lib.utils import birdfile2class, birdfile2idx, is_test_bird_idx, get_bird_bbox, get_bird_class, get_bird_part, get_part_location, get_multi_part_location, get_bird_name
from lib.utils import get_attribute_name, code2certainty, get_class_attributes, get_image_attributes, describe_bird, attribute2idx


class_attr = True # use class level attributes
cub = CUB(class_attr)

train_val_indices = [i for i in range(len(cub)) if not is_test_bird_idx(birdfile2idx(cub.images_path[i]))]
train_val_labels = [cub.labels[i] for i in range(len(cub)) if not is_test_bird_idx(birdfile2idx(cub.images_path[i]))]
val_ratio = 0.2
train_indices, val_indices = train_test_split(train_val_indices, test_size=val_ratio, stratify=train_val_labels)
test_indices = [i for i in range(len(cub)) if is_test_bird_idx(birdfile2idx(cub.images_path[i]))]
cub_train = CUB_train_transform(Subset(cub, train_indices))
cub_val = CUB_test_transform(Subset(cub, val_indices))
cub_test = CUB_test_transform(Subset(cub, test_indices))

# accuracy
acc_criterion = lambda o, y: (o.argmax(1) == y).float()

# define concepts to learn
class_attributes = get_class_attributes()
maj_concepts = class_attributes.loc[:, ((class_attributes >= 50).sum(0) >= 10)] >= 50 # CBM paper report 112 concepts; here is 108
ind_maj_attr = set(maj_concepts.columns) 
print(f'ind_maj_attr has {len(ind_maj_attr)} concepts to learn')

def concept_model(name, loader_xy, loader_xy_val, loader_xy_te, n_epochs=10,
                  report_every=1, plot=False, device='cuda'):
    # regular model
    net = torch.hub.load('pytorch/vision:v0.9.0', 'inception_v3', pretrained=True)
    net.fc = nn.Linear(2048, 2) # concept binary model
    net.to(device)
    print('task acc before training: {:.1f}%'.format(test(net,
                                                          loader_xy, acc_criterion,
                                                          device=device) * 100))
    
    # train
    # opt = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    opt = optim.Adam(net.parameters())
    log = train(net, loader_xy, opt, n_epochs=n_epochs, report_every=report_every,
                device=device, savepath=f"{RootPath}/models/{name}",
                report_dict={'val acc': (lambda m: test(m, loader_xy_val, acc_criterion,
                                                        device=device) * 100, 'max'),
                             'train acc': (lambda m: test(m, loader_xy, acc_criterion,
                                                          device=device) * 100, 'max'),
                             'test acc': (lambda m: test(m, loader_xy_te, acc_criterion,
                                                         device=device) * 100, 'max')},
                early_stop_metric='val acc')
    
    if plot: plot_log(log)
    print('task acc after training: {:.1f}%'.format(test(net, loader_xy_te,
                                                         acc_criterion,
                                                         device=device) * 100))
    return net

# todo: learn a single model instead
models = []
for attr in ind_maj_attr:
    print(attr)
    loader_xy_ = DataLoader(SubColumn(SubAttr(cub_train, attr), ['x', 'attr']),
                            batch_size=32, shuffle=True, num_workers=8)
    loader_xy_val_ = DataLoader(SubColumn(SubAttr(cub_val, attr), ['x', 'attr']),
                                batch_size=32, shuffle=False, num_workers=8)    
    loader_xy_te_ = DataLoader(SubColumn(SubAttr(cub_test, attr), ['x', 'attr']),
                               batch_size=32, shuffle=False, num_workers=8)
    concept_model(f"attr_{int(class_attr)}_{attr}", loader_xy_, loader_xy_val_,
                  loader_xy_te_, n_epochs=10, report_every=1)
