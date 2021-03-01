# Base data manipulation stack
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

import pytorch_lightning as pl

from sklearn.model_selection import StratifiedKFold, train_test_split

# utils from src
try:
    from .config import Config
except ImportError:
    from config import Config


# seed
pl.seed_everything(Config.seed_val)


class GraphemeDataset(Dataset):
    def __init__(self, df: pd.DataFrame, task: str = 'train', images_dir: str = Config.images_dir, transform=None):
        super(GraphemeDataset, self).__init__()
        self.df = df
        self.task = task
        self.images_dir = images_dir
        self.transform = transform

    def __getitem__(self, index):
        image_id = self.df.iloc[index].image_id

        # load raw image & normalize
        arr = np.load(os.path.join(self.images_dir, image_id+'.npy')) / 255.
        img = Image.fromarray(arr)
        if self.transform is not None:
            try:
                img = self.transform(image=img)['image']
                # setup sample dict
                sample = {
                    'image': img
                }
            except:
                img = self.transform(img)
                img = np.array(img)
                sample = {
                    'image': th.from_numpy(img).float()
                }

        else:
            # setup sample dict
            sample = {
                'image': th.from_numpy(np.array(img)).float()
            }

        if self.task == 'train':
            grapheme_root = self.df.iloc[index].grapheme_root
            vowel_diacritic = self.df.iloc[index].vowel_diacritic
            consonant_diacritic = self.df.iloc[index].consonant_diacritic
            grapheme = self.df.iloc[index].grapheme

            sample.update({
                'grapheme_root':  th.tensor(grapheme_root, dtype=th.long),
                'vowel_diacritic':  th.tensor(vowel_diacritic, dtype=th.long),
                'consonant_diacritic':  th.tensor(consonant_diacritic, dtype=th.long),
                'grapheme':  str(grapheme)
            })
        return sample

    def __len__(self):
        return len(self.df)


class DataModule(pl.LightningDataModule):
    def __init__(self, df: pd.DataFrame, frac=1, validation_split=.25, train_batch_size=4, test_batch_size=4, transform: dict = None):
        super(DataModule, self).__init__()
        self.frac = frac
        self.df = df
        self.train_bs = train_batch_size
        self.test_bs = test_batch_size
        self.validation_split = validation_split
        self.transform = transform

    def setup(self):
        data = self.df.sample(frac=self.frac).reset_index(drop=True)
        train_set, val_set = train_test_split(
            data, test_size=self.validation_split)

        print(f"[INFO] Training on {len(train_set)} samples")

        self.train_dataset = GraphemeDataset(
            df=train_set,
            images_dir=Config.images_dir,
            task='train',
            transform=self.transform['train']
        )

        if len(val_set) > 0:
            print(f"[INFO] Validating on {len(val_set)} samples")

            self.valid_dataset = GraphemeDataset(
                df=val_set,
                images_dir=Config.images_dir,
                task='train',
                transform=self.transform['validation']
            )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.train_bs,
            num_workers=Config.num_workers,
            shuffle=True
        )

    def test_dataloader(self):
        pass

    def val_dataloader(self):
        return DataLoader(
            dataset=self.valid_dataset,
            batch_size=self.test_bs,
            num_workers=Config.num_workers,
            shuffle=False
        )


if __name__ == '__main__':
    pass
