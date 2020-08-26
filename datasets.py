from torch.utils.data import Dataset
from PIL import Image
import random
import glob
import os


# Define the dataset class
# In refrence to the following implementation: https://github.com/aitorzip/PyTorch-CycleGAN.git
class ImageDataset(Dataset):
    def __init__(self, root, transform = None, mode = 'train'):
        self.transform = transform
        self.files_A = sorted(glob.glob(os.path.join(root, '%s/A' % mode) + '/*.*'))
        self.mode = mode
        if self.mode == 'train':
            self.files_B = sorted(glob.glob(os.path.join(root, '%s/B' % mode) + '/*.*'))


    def __getitem__(self, index):
        item_A = self.transform(Image.open(self.files_A[index % len(self.files_A)]))
        if self.mode == 'train':
            item_B = self.transform(Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)]))
            return {'A': item_A, 'B': item_B}
        else:
            return self.files_A[index % len(self.files_A)], item_A

    def __len__(self):
        if self.mode == 'train':
            return max(len(self.files_A), len(self.files_B))
        else:
            return len(self.files_A)


# Define the paired dataset class
class PairedImage(Dataset):
    def __init__(self, root, transform = None, mode = 'train'):
        self.transform = transform
        self.files_A = sorted(glob.glob(os.path.join(root, '%s/A' % mode) + '/*.*'))
        self.mode = mode
        if self.mode == 'train':
            self.files_B = sorted(glob.glob(os.path.join(root, '%s/B' % mode) + '/*.*'))

    def __getitem__(self, index):
        item_A = self.transform(Image.open(self.files_A[index]))
        if self.mode == 'train':
            item_B = self.transform(Image.open(self.files_B[index]))
            return {'A': item_A, 'B': item_B}
        else:
            return self.files_A[index], item_A

    def __len__(self):
        return len(self.files_A)
