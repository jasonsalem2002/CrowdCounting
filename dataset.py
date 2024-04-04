import torch
from torch.utils.data import Dataset
import h5py
import glob
import os
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
from torchvision.transforms import functional as F
import torchvision

class ShanghaiTechDataset(Dataset):
    def __init__(self, root, part='part_A', phase='train', transform=None):
        self.root = os.path.join(root, part, f"{phase}_data")
        # print("ShanghaiTechDataset root: ", self.root)
        self.image_paths = glob.glob(os.path.join(self.root, 'images', '*.jpg'))
        # print("ShanghaiTechDataset image_paths: ", self.image_paths)
        self.gt_paths = [p.replace('.jpg', '.h5').replace('images', 'ground-truth') for p in self.image_paths]
        # print("ShanghaiTechDataset gt_paths:",self.gt_paths)
        self.transform = transform
        # print("ShanghaiTechDataset transform: ", self.transform)
        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
            ])
        else:
            self.transform = transform

    def __getitem__(self, index):
        # Load image
        img = Image.open(self.image_paths[index]).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        # Load density map
        gt_file = h5py.File(self.gt_paths[index], 'r')
        target = np.asarray(gt_file['density'])

        # Convert density map to tensor
        target = torch.from_numpy(target).float()

        # Ensure output is CxHxW
        if len(target.shape) == 2:
            target = target.unsqueeze(0)

        target = F.resize(target, size=(96, 128), interpolation=torchvision.transforms.InterpolationMode.BILINEAR)

        return img, target

    def __len__(self):
        return len(self.image_paths)
