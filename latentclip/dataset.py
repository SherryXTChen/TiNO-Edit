import os
import glob
from PIL import Image
from torch.utils.data import Dataset

from utils import pil_to_tensor

class ImageDataset(Dataset):
    def __init__(self, root, split):
        self.image_list = []
        for d in os.listdir(root):
            self.image_list += glob.glob(f'{root}/{d}/*.jpg')

        if split == 'train':
            self.image_list = self.image_list[:int(0.8*len(self.image_list))]
        else:
            self.image_list = self.image_list[int(0.8*len(self.image_list)):]

    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, index):
        image = Image.open(self.image_list[index]).convert('RGB')
        tensor = pil_to_tensor(image)[0]
        return tensor