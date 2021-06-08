import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
from torchvision import transforms
import os
import pathlib
from torchvision.transforms import GaussianBlur, CenterCrop, ColorJitter, Grayscale, RandomCrop, RandomHorizontalFlip

# custom
from lib.utils import birdfile2class, attribute2idx, get_class_attributes
#, birdfile2idx, is_test_bird_idx, get_bird_bbox, get_bird_class, get_bird_part, get_part_location, get_multi_part_location, get_bird_name
#  from lib.utils import get_attribute_name, code2certainty, get_class_attributes, get_image_attributes

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
    '''

    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        d = self.dataset[idx]
        d['x'] = self.transform(d['x'])
        return d
    
########### CUB specific
class SubAttr(Dataset):
    '''
    a dataset taking sub attributes
    for learning each concept
    '''
    def __init__(self, dataset, attr_name):
        self.dataset = dataset
        self.attr_idx = attribute2idx(attr_name)-1 # 0-index

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        d = self.dataset[idx]
        d['attr'] = d['attr'][self.attr_idx].long()
        return d

def CUB_train_transform(dataset):
    '''
    transform dataset according to the CBM paper
    '''
    resol = 299
    transform = transforms.Compose([
        transforms.ColorJitter(brightness=32/255, saturation=(0.5, 1.5)),
        transforms.RandomResizedCrop(resol),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), #implicitly divides by 255
        transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [2, 2, 2])
        # transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ], std = [ 0.229, 0.224, 0.225 ]), # imagenet setting
    ])

    return TransformWrapper(dataset, transform)

def CUB_test_transform(dataset):
    '''
    transform dataset according to the CBM paper
    todo: check the exact paper setting
    '''
    resol = 299
    transform = transforms.Compose([
        transforms.CenterCrop(resol),
        transforms.ToTensor(), #implicitly divides by 255
        transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [2, 2, 2])
        #transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ], std = [ 0.229, 0.224, 0.225 ]),
    ])
        
    return TransformWrapper(dataset, transform)

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

    def __init__(self):
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

        
    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, idx):
        # based on example here: https://pytorch.org/hub/pytorch_vision_inception_v3/
        filename = self.images_path[idx]
        x = Image.open(filename)
        # preprocess = transforms.Compose([ # for imagenet
        #         transforms.Resize(299),
        #         transforms.CenterCrop(299),
        #         transforms.ToTensor(),
        #         transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                              std=[0.229, 0.224, 0.225]),
        #     ])
        x, y = x, self.labels[idx]        
        
        return {"x": x, "y": y, "filename": filename,
                "attr": self.class_attributes[y]}

