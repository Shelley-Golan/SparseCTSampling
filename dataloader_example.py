import os
import random
import io
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
from torchvision import transforms
from tqdm import tqdm
import pydicom as dicom
#from pydicom.pixel_data_handlers.util import apply_voi_lut
from PIL import Image
import cv2

class CTDataset(Dataset):
    def __init__(self, args, task: str = 'train', sparsity: int = 16, theta = None):
        """
        Args:
            args:
            task: Select the dataset to load from (train/val/test).
        """
        self.task = task
        self.normalize = args.normalize
        images = []
        self.path = f'{args.dataset_path}/{task}'
        self.df = os.listdir(self.path)
        '''
        transform = transforms.ToTensor()
        for i, name in enumerate(os.listdir(self.path)):
            file = os.path.join(self.path, name)
            print(file)
            ds = dicom.dcmread(file)
            if "PixelData" not in ds:
                print(f"Dataset found with no Pixel Data: {file}")
                continue

            data = ds.pixel_array
            data = data - np.min(data)
            data = data / np.max(data)
            #sample = (data * 255).astype(np.uint8)
            sample = data
            images +=[transform(sample).float()]
        self.df = torch.cat(images, dim=0).unsqueeze(1)
        '''
        # collect samples
        self.examples = np.arange(len(self.df))
        if args.sample_rate < 1:
            random.shuffle(self.examples)
            num_files = round(len(self.examples) * args.sample_rate)
            self.examples = self.examples[:num_files]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        index = self.examples[idx]
        name = self.df[index]
        transform = transforms.ToTensor()
        file = os.path.join(self.path, name)
        ds = dicom.dcmread(file)
        if "PixelData" not in ds:
            print(f"Dataset found with no Pixel Data: {file}")
        data = ds.pixel_array
        data = data - np.min(data)
        data = data / np.max(data)
        # sample = (data * 255).astype(np.uint8)
        sample = transform(data).float()
        #if self.normalize:
        #    mean = down_sampled.mean(dim=(2,3), keepdims=True)
        #    std = down_sampled.std(dim=(2,3), keepdims=True)
        #    down_sampled = (down_sampled-mean)/std
        return sample

def create_data_loaders(args):
    train_dataset = CTDataset(
        args=args,
        task='train',
    )
    val_dataset = CTDataset(
        args=args,
        task='val',
    )
    test_dataset = CTDataset(
        args=args,
        task='test',
    )
    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=args.num_workers,
                              pin_memory=True)

    val_loader = DataLoader(val_dataset,
                            batch_size=args.batch_size,
                            shuffle=False,
                            num_workers=args.num_workers,
                            pin_memory=True)

    test_loader = DataLoader(test_dataset,
                             batch_size=args.batch_size,
                             shuffle=False,
                             num_workers=args.num_workers,
                             pin_memory=True)
    return train_loader, val_loader, test_loader
