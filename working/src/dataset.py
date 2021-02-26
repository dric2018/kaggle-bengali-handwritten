# Base data manipulation stack
from .config import Config
import os
import numpy as np
import pandas as pd
from PIL import Image

# DL stack
import torch as th 
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models
from torchvision import transforms   
import albumentations as alb

import pytorch_lightning as pl

# utils from src
from .config import Config


class BengaliDataset(Dataset):
    def __init__(self, df:pd.DataFrame, task:str='train', data_dir:str=Config.data_dir, transform=None):
        super(BengaliDataset, self).__init__()
        self.df = df
        self.task = task
        self.data_dir = data_dir
        self.transform = transform

    def __getitem__(self, index):
        image_id = self.df.iloc[index].image_id

        # load raw image
        img = Image.open(os.path.join(self.data_dir, image_id))
        if self.transform is not None:
            try:
                img = transform(image=img)['image']
                # setup sample dict
                sample = {
                    'image' : img
                }
            except:
                # setup sample dict
                sample = {
                    'image' : th.tensor(img, dtype=th.float)
                }

        

        if self.task == 'train':
            grapheme_root = self.df.iloc[index].grapheme_root
            vowel_diacritic = self.df.iloc[index].vowel_diacritic
            constant_diacritic = self.df.iloc[index].constant_diacritic
            grapheme = self.df.iloc[index].grapheme

            sample.update({
                'grapheme_root' :  th.tensor(grapheme_root, dtype=th.long),
                'vowel_diacritic' :  th.tensor(vowel_diacritic, dtype=th.long),
                'constant_diacritic' :  th.tensor(constant_diacritic, dtype=th.long),
                'grapheme' :  str(grapheme)
            })
        return sample


    def __len__(self):
        return len(self.df)


class DataModule(pl.LightningDataModule):
    pass


if __name__ == '__main__':
    pass
