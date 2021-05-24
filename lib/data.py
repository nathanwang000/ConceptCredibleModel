import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
from torchvision import transforms
import os

class small_CUB(Dataset):
    '''
    small CUB dataset just for thesis proposal
    '''

    def __init__(self, transform=lambda x, y: x):
        self.bird1_dir = "datasets/small_bird_data/001.Black_footed_Albatross/"
        self.bird2_dir = "datasets/small_bird_data/002.Laysan_Albatross/"
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

    
