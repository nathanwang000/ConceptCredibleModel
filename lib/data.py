import torch
from collections.abc import Iterable
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
from torchvision import transforms
import os
import pathlib
import tqdm
from torchvision.transforms import GaussianBlur, CenterCrop, ColorJitter, Grayscale, RandomCrop, RandomHorizontalFlip
# torch.multiprocessing.set_start_method('spawn') # only useful when cuda need inside transform

# custom
from lib.utils import birdfile2class, attribute2idx, get_class_attributes, get_image_attributes, get_attribute_name, birdfile2idx, get_shortcut_level

class SubColumn(Dataset):
    '''
    a dataset that select sub columns of another dataset, useful for dataset in which not all columns 
    need to be extracted: e.g., in CUB dataset sometimes we just need x, y information
    '''

    def __init__(self, dataset, column_names):
        self.dataset = dataset
        self.column_names = column_names

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        d = self.dataset[idx]
        return tuple(d[c] for c in self.column_names)

class TransformWrapper(Dataset):
    '''
    add transformation on the dataset (useful for different transformation for train test)
    this is a internal wrapper meant only for this file
    '''

    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        d = self.dataset[idx]
        d = self.transform(d)
        return d

        
########### CUB specific
class SubAttr(Dataset):
    '''
    a dataset taking sub attributes
    for learning each concept
    '''
    def __init__(self, dataset, attr_names):
        self.dataset = dataset
        self.attr_indices = list(map(lambda attr: attribute2idx(attr) - 1,
                                     attr_names)) # 0 index

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        d = self.dataset[idx]
        if len(self.attr_indices) == 1:
            d['attr'] = d['attr'][self.attr_indices[0]].long()
        else:
            d['attr'] = d['attr'][self.attr_indices].long()
        return d

def x_transform(transform):
    '''
    d is a dataset instance: e.g. datset[idx]
    applies transform on d['x'] and returns d
    '''
    def _transform(d):
        d['x'] = transform(d['x'])
        return d
    
    return _transform

def CUB_train_transform(dataset, mode="cbm"):
    '''
    transform dataset according to the CBM paper
    mode: cbm or custom
    '''
    resol = 299
    # input transform
    if mode == "cbm":
        transform = transforms.Compose([
            transforms.ColorJitter(brightness=32/255, saturation=(0.5, 1.5)),
            transforms.RandomResizedCrop(resol),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(), #implicitly divides by 255
            transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [2, 2, 2])
        ])
    elif mode == 'imagenet': # same train test but use imagenet trainsform 77.9% acc
        transform = transforms.Compose([ # for imagenet
                transforms.Resize(299), # this only resize the smaller size
                transforms.CenterCrop(299),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])
    elif mode == 'same': # 77.7% acc; only center crop is ok
        transform = transforms.Compose([
            transforms.CenterCrop(299), # this makes sure it is centered
            transforms.ToTensor(),
            transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [2, 2, 2])
            ])
    elif mode == 'flip':
        transform = transforms.Compose([
            transforms.CenterCrop(299), # this makes sure it is centered
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(), #implicitly divides by 255
            transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [2, 2, 2])
        ])
    elif mode == 'crop':
        transform = transforms.Compose([
            transforms.RandomResizedCrop(resol),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(), #implicitly divides by 255
            transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [2, 2, 2])
        ])
    
    return TransformWrapper(dataset, x_transform(transform))

def CUB_test_transform(dataset, mode="cbm"):
    '''
    transform dataset according to the CBM paper
    '''
    if mode == 'imagenet':
        transform = transforms.Compose([ # for imagenet
            transforms.Resize(299), # this only trims the smaller side
            transforms.CenterCrop(299), # this makes sure it is centered
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
            ])
    else:
        transform = transforms.Compose([
            transforms.CenterCrop(299), # this makes sure it is centered
            transforms.ToTensor(), # implicitly divides by 255
            transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [2, 2, 2])
            ])
        
    return TransformWrapper(dataset, x_transform(transform))

class small_CUB(Dataset):
    '''
    small CUB dataset just for thesis proposal
    Caltech UCSD bird dataset http://www.vision.caltech.edu/visipedia/papers/CUB_200_2011.pdf
    based on example here: https://pytorch.org/hub/pytorch_vision_inception_v3/
    '''

    def __init__(self, transform=lambda x, y: x):
        pwd = pathlib.Path(__file__).parent.absolute()
        self.bird1_dir = f"{pwd}/../datasets/small_bird_data/001.Black_footed_Albatross/"
        self.bird2_dir = f"{pwd}/../datasets/small_bird_data/002.Laysan_Albatross/"
        a = [os.path.join(os.path.dirname(self.bird1_dir), i) \
             for i in os.listdir(self.bird1_dir) if i[-3:]=="jpg"]
        b = [os.path.join(os.path.dirname(self.bird2_dir), i) \
             for i in os.listdir(self.bird2_dir) if i[-3:]=="jpg"]
        self.images_path = a + b
        self.labels = [0] * len(a) + [1] * len(b)
        self.transform = transform

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, idx):
        filename = self.images_path[idx]
        input_image = Image.open(filename)
        preprocess = transforms.Compose([
                transforms.Resize(299),
                transforms.CenterCrop(299),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])
        x, y = preprocess(input_image), self.labels[idx]
        return self.transform(x, y), y
    
class CUB(Dataset):
    '''
    Caltech UCSD bird dataset http://www.vision.caltech.edu/visipedia/papers/CUB_200_2011.pdf
    '''

    def __init__(self, class_attr=True):
        '''
        class_attr: if True use class level attributes, else use image level attributes
        '''
        self.class_attr = class_attr
        # ignore grayscale images (computed from birds_gray.ipynb)
        gray_ims = ['Clark_Nutcracker_0020_85099.jpg',
                    'Pelagic_Cormorant_0022_23802.jpg',
                    'Mallard_0130_76836.jpg',
                    'White_Necked_Raven_0070_102645.jpg',
                    'Western_Gull_0002_54825.jpg',
                    'Brewer_Blackbird_0028_2682.jpg',
                    'Ivory_Gull_0040_49180.jpg',
                    'Ivory_Gull_0085_49456.jpg']
        
        pwd = pathlib.Path(__file__).parent.absolute()
        self.dir = f"{pwd}/../datasets/bird_data/CUB_200_2011/images/"
        self.images_path = [os.path.join(os.path.dirname(self.dir), image_path, i) \
                            for image_path in os.listdir(self.dir) \
                            for i in os.listdir(os.path.join(os.path.dirname(self.dir),
                                                             image_path)) if i[-3:]=="jpg" and i not in gray_ims]
        
        self.labels = [birdfile2class(fn)-1 for fn in self.images_path] # -1 b/c 1 indexed

        class_attributes = get_class_attributes() >= 50
        self.class_attributes = torch.from_numpy(np.array(class_attributes)).float()

        # individual attributes
        if not os.path.exists(f"{pwd}/../datasets/bird_data/attributes.pkl"):
            print("reading image level attributes (~30s)")
            image_attributes = get_image_attributes() # 30s
            pivot_image_attributes = image_attributes.replace({"attr_idx": dict((i, get_attribute_name(i)) for i in range(1, 313))}).pivot(index='image_idx', columns='attr_idx')
            torch.save(pivot_image_attributes, f"{pwd}/../datasets/bird_data/attributes.pkl")
        else:
            pivot_image_attributes = torch.load(f"{pwd}/../datasets/bird_data/attributes.pkl")
        self.image_attributes = torch.from_numpy(np.array(pivot_image_attributes['is_present'].\
            loc[:, [get_attribute_name(i) for i in range(1, 313)]].\
            loc[[birdfile2idx(fn) for fn in self.images_path]])).float() # sort by image id
        
    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, idx):
        # based on example here: https://pytorch.org/hub/pytorch_vision_inception_v3/
        filename = self.images_path[idx]
        x = Image.open(filename)
        x, y = x, self.labels[idx]
        if self.class_attr:
            attr = self.class_attributes[y]
        else:
            attr = self.image_attributes[idx]
        return {"x": x, "y": y, "filename": filename,
                "attr": attr}

###### shortcut transforms
def shortcut_noise_transform(x, y, n_shortcuts, sigma_max=0.1, threshold=0,
                             shortcut_level=None):
    '''
    x: input to transform (bs, *) or (*)
    y: noise depend on this variable (bs,) or int or float
    n_shortcuts: number of shortcut classes
    sigma_max: max noise scale, threshold: before independent noise
    shortcut_level: manual shortcut level for debug
    return transformed x
    '''
    assert n_shortcuts <= 200, "shortcut classes <= 200"

    sigmas = torch.linspace(0, sigma_max, n_shortcuts)

    if shortcut_level is None:
        shortcut_level = get_shortcut_level(y, threshold, n_shortcuts)
            
    with torch.no_grad():
        sigma = sigmas[shortcut_level] # (bs,) or 1
        
    if isinstance(y, Iterable):
        # batch version
        x = x + (sigma.to(y.device) * torch.randn_like(x).transpose(0, -1))\
            .transpose(0, -1)
    else:
        x = x + sigma * torch.randn_like(x)
    return x, shortcut_level

def CUB_shortcut_transform(x, y, **kwargs):
    if 'shortcut_mode' not in kwargs: return x
    mode = kwargs['shortcut_mode']
    if mode == 'clean':
        return x, torch.zeros(x.shape[0]).to(x.device)
    elif mode == 'noise':
        x, s = shortcut_noise_transform(x, y,
                                        n_shortcuts=kwargs['n_shortcuts'],
                                        threshold=kwargs['shortcut_threshold'])
    else:
        x, s = shortcut_noise_transform(x, kwargs['net_shortcut'](x).argmax(1),
                                        n_shortcuts=kwargs['n_shortcuts'],
                                        threshold=kwargs['shortcut_threshold'])
    return x, s
    
