import pathlib
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import matplotlib
import numpy as np

def dfs_freeze(model):
    ''' 
    freeze all paramters of a pytorch model
    '''
    for name, child in model.named_children():
        for param in child.parameters():
            param.requires_grad = False
        dfs_freeze(child)
                                            
##### CUB specific utilities
def get_attr_names(fn):
    # attributes to use
    attr_names = []
    with open(fn, "r") as f:
        for l in f:
            attr_names.append(l.strip())
    return attr_names

def birdfile2class(filename):
    return int(filename.split("/")[-2][:3])

def code2certainty(code):
    pwd = pathlib.Path(__file__).parent.absolute()
    code2name = {}
    with open(f'{pwd}/../datasets/bird_data/CUB_200_2011/attributes/certainties.txt') as f:
        for l in f:
            parts = l.split()
            code2name[int(parts[0])] = " ".join(parts[1:])
            
    return code2name[int(code)]


def get_image_attributes():
    # n x n_attribute
    pwd = pathlib.Path(__file__).parent.absolute()    
    ans = []
    with open(f'{pwd}/../datasets/bird_data/CUB_200_2011/attributes/image_attribute_labels.txt') as f:
        for l in f:
            ans.append(list(map(lambda x: x[0](x[1]),
                                zip([int, int, int, int, float],
                                    l.split()))))
    ans = pd.DataFrame(ans, columns=['image_idx',  'attr_idx', 'is_present', 'certainty', 'time'])
    return ans

def get_class_attributes():
    pwd = pathlib.Path(__file__).parent.absolute()    
    ans = []
    with open(f'{pwd}/../datasets/bird_data/CUB_200_2011/attributes/class_attribute_labels_continuous.txt') as f:
        for l in f:
            ans.append(list(map(lambda x: float(x), l.split())))
    ans = pd.DataFrame(ans, columns=[get_attribute_name(i) for i in range(1, 313)])
    return ans
                                            
def build_birdfile2idx():
    pwd = pathlib.Path(__file__).parent.absolute()
    bird2idx = {}
    with open(f'{pwd}/../datasets/bird_data/CUB_200_2011/images.txt') as f:
        for l in f:
            idx, bird = l.split()
            bird2idx[bird] = int(idx)

    def _f(filename):
        return bird2idx['/'.join(filename.split('/')[-2:])]

    return _f

birdfile2idx = build_birdfile2idx()

def build_get_bird_name():
    pwd = pathlib.Path(__file__).parent.absolute()
    idx2name = {}
    with open(f'{pwd}/../datasets/bird_data/CUB_200_2011/classes.txt') as f:
        for l in f:
            idx, name = l.split()
            idx2name[int(idx)] = name
            
    def _f(class_idx):
         return idx2name[int(class_idx)]

    return _f

get_bird_name = build_get_bird_name()

def build_attribute2idx():
    pwd = pathlib.Path(__file__).parent.absolute()
    name2idx = {}
    with open(f'{pwd}/../datasets/bird_data/attributes.txt') as f:
        for l in f:
            idx, name = l.split()
            name2idx[name] = int(idx)
            
    def _f(name):
         return name2idx[name]

    return _f

attribute2idx = build_attribute2idx()

def build_get_attribute_name():
    pwd = pathlib.Path(__file__).parent.absolute()
    idx2name = {}
    with open(f'{pwd}/../datasets/bird_data/attributes.txt') as f:
        for l in f:
            idx, name = l.split()
            idx2name[int(idx)] = name
            
    def _f(attr_idx):
         return idx2name[int(attr_idx)]

    return _f

get_attribute_name = build_get_attribute_name()

def build_get_bird_part():
    pwd = pathlib.Path(__file__).parent.absolute()
    idx2part = {}
    with open(f'{pwd}/../datasets/bird_data/CUB_200_2011/parts/parts.txt') as f:
        for l in f:
            parts = l.split()
            idx2part[int(parts[0])] = " ".join(parts[1:])
            
    def _f(query_part_idx):
         return idx2part[int(query_part_idx)]

    return _f

get_bird_part = build_get_bird_part()

def build_get_part_location():
    pwd = pathlib.Path(__file__).parent.absolute()
    idx2part = {}
    with open(f'{pwd}/../datasets/bird_data/CUB_200_2011/parts/part_locs.txt') as f:
        for l in f:
            image_idx, part_idx, x, y, visible = list(map(lambda x: x[0](x[1]),
                                                          zip([int, int, float, float, int], l.split())))
            if image_idx not in idx2part:
                idx2part[image_idx] = {}
            idx2part[image_idx][part_idx] = (x, y, visible)

    def _f(query_image_idx):
        return idx2part[int(query_image_idx)]

    return _f

get_part_location = build_get_part_location()

def build_get_multi_part_location():
    '''multiple annotator location'''
    pwd = pathlib.Path(__file__).parent.absolute()
    idx2part = {}
    with open(f'{pwd}/../datasets/bird_data/CUB_200_2011/parts/part_click_locs.txt') as f:
        for l in f:
            image_idx, part_idx, x, y, visible, label_time = list(map(lambda x: x[0](x[1]),
                                                                      zip([int, int, float, float, int, float], l.split())))
            if image_idx not in idx2part:
                idx2part[image_idx] = {}
            if part_idx not in idx2part[image_idx]:
                idx2part[image_idx][part_idx] = []
            idx2part[image_idx][part_idx].append((x, y, visible, label_time))

    def _f(query_image_idx):
        return idx2part[int(query_image_idx)]

    return _f

get_multi_part_location = build_get_multi_part_location()

def build_is_test_bird_idx():
    pwd = pathlib.Path(__file__).parent.absolute()
    is_test = {}
    with open(f'{pwd}/../datasets/bird_data/CUB_200_2011/train_test_split.txt') as f:
        for l in f:
            idx, is_train = l.split()
            is_test[int(idx)] = 1 - int(is_train)

    def _f(query_idx):
        return is_test[int(query_idx)]

    return _f

is_test_bird_idx = build_is_test_bird_idx()

def build_get_bird_bbox():
    pwd = pathlib.Path(__file__).parent.absolute()
    idx2bbox = {} # x, y, width, height
    with open(f'{pwd}/../datasets/bird_data/CUB_200_2011/bounding_boxes.txt') as f:
        for l in f:
            idx, x, y, w, h = l.split()
            idx2bbox[int(idx)] = (float(x), float(y), float(w), float(h))

    def _f(query_idx):
        return idx2bbox[int(query_idx)]

    return _f

get_bird_bbox = build_get_bird_bbox()

def build_get_bird_class():
    pwd = pathlib.Path(__file__).parent.absolute()
    idx2class = {} # x, y, width, height
    with open(f'{pwd}/../datasets/bird_data/CUB_200_2011/image_class_labels.txt') as f:
        for l in f:
            idx, cls = l.split()
            idx2class[int(idx)] = int(cls)

    def _f(query_idx):
        return idx2class[int(query_idx)]

    return _f

get_bird_class = build_get_bird_class()

def describe_bird(filename):
    print(f"filename: {filename}")
    bird_idx = birdfile2idx(filename)
    print(f"image id: {bird_idx}")
    class_idx = get_bird_class(bird_idx) # birdfile2class(filename) # 2 ways
    print(f"class id: {class_idx}")
    print(f"bird name: {get_bird_name(class_idx)}")
    print(f"is_test: {is_test_bird_idx(bird_idx)}")

    ##########################
    ## show image
    im = Image.open(filename)
    print("image size:", np.array(im).shape)
    # Create figure and axes
    fig, ax = plt.subplots()
    # Display the image
    ax.imshow(im)

    ## bounding box
    x, y, w, h = get_bird_bbox(birdfile2idx(filename))
    rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)

    ## parts annotation
    cmap = matplotlib.cm.get_cmap('tab20')
    part_locs = get_part_location(bird_idx)
    for part_idx, (x, y, v) in part_locs.items():
        if v: # visible
            circle = patches.Circle((x, y), 2, color=cmap(part_idx/len(part_locs)), label=get_bird_part(part_idx))
            ax.add_patch(circle)

    ## show image
    plt.legend(bbox_to_anchor=(1,1))
    plt.axis('off')
    plt.show()

    ##########################
    ## show image
    im = Image.open(filename)
    # Create figure and axes
    fig, ax = plt.subplots()
    # Display the image
    ax.imshow(im)

    ## multiple parts annotation by different annotators
    cmap = matplotlib.cm.get_cmap('tab20')
    part_multi_locs = get_multi_part_location(bird_idx)
    for part_idx, l in part_multi_locs.items():
        for x, y, v, t in l:
            if v: # visible
                circle = patches.Circle((x, y), 2, color=cmap(part_idx/len(part_multi_locs)), label=get_bird_part(part_idx))
                ax.add_patch(circle)

    ## show image
    plt.axis('off')
    plt.show()
    
