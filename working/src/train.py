import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.metrics.functional import recall
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, GPUStatsMonitor
from pytorch_lightning.loggers import TensorBoardLogger


from tqdm.auto import tqdm
from sklearn.utils.class_weight import compute_class_weight

import numpy as np
import pandas as pd

import os
import gc
try:
    from .config import Config
    from .dataset import DataModule, GraphemeDataset
    from .model import GraphemeClassifier

    from .utils import train_model, run_kFold
except ImportError:
    from config import Config
    from dataset import DataModule, GraphemeDataset
    from model import GraphemeClassifier

    from utils import train_model, run_kFold


if __name__ == '__main__':

    print('[INFO] Building dataset/datamodule')
    # get dataset/dataloader
    train_df = pd.read_csv(os.path.join(Config.data_dir, 'train.csv'))

    if Config.data_transform == "basic":
        data_transforms = {
            'train': th.nn.Sequential(
                transforms.CenterCrop(Config.resize_shape),
                transforms.RandomRotation(
                    degrees=65, resample=False, expand=False, center=None, fill=0.0),
                transforms.RandomVerticalFlip(p=0.04),
                transforms.RandomHorizontalFlip(p=0.6),
            ),
            "validation": th.nn.Sequential(
                transforms.CenterCrop(Config.resize_shape),
                transforms.RandomRotation(
                    degrees=35, resample=False, expand=False, center=None, fill=0.0),
                transforms.RandomHorizontalFlip(p=0.6),
            ),
            'test': th.nn.Sequential(
                transforms.CenterCrop(Config.resize_shape),

            )
        }

    dm = DataModule(
        df=train_df,
        frac=1,
        validation_split=.3,
        train_batch_size=Config.train_batch_size,
        test_batch_size=Config.test_batch_size,
        transform=data_transforms
    )

    dm.setup()

    # define training pipeline
    # classes weights
    print('[INFO] Computing classes weights')
    vowels_class_weight = compute_class_weight(
        class_weight='balanced',
        classes=train_df.vowel_diacritic.unique(),
        y=train_df.vowel_diacritic.values
    )
    g_root_class_weight = compute_class_weight(
        class_weight='balanced',
        classes=train_df.grapheme_root.unique(),
        y=train_df.grapheme_root.values
    )
    consonant_class_weight = compute_class_weight(
        class_weight='balanced',
        classes=train_df.consonant_diacritic.unique(),
        y=train_df.consonant_diacritic.values
    )

    print(f'[INFO] Defining Model based on {Config.base_model} architecture')
    # define model/model architectures
    model = GraphemeClassifier(
        base_encoder=Config.base_model,
        arch_from='timm',
        vowels_class_weight=th.from_numpy(vowels_class_weight).float(),
        g_root_class_weight=th.from_numpy(g_root_class_weight).float(),
        consonant_class_weight=th.from_numpy(consonant_class_weight).float(),
        drop=0.25,
        lr=Config.learning_rate,
        pretrained=True
    )
    print(f'[INFO] Defining Callbacks')
    # callbacks

    model_ckpt = ModelCheckpoint(
        filename=os.path.join(
            Config.models_dir, f"bengali_grapheme-{Config.base_model}"),
        monitor='val_recall',
        mode="max"
    )
    es = EarlyStopping(
        monitor='val_recall',
        patience=15,
        mode="max"
    )
    gpu_stats = GPUStatsMonitor(
        memory_utilization=True,
        gpu_utilization=True,
        intra_step_time=False,
        inter_step_time=False,
        fan_speed=True,
        temperature=True,
    )

    callbacks_list = [es, model_ckpt, gpu_stats]

    # logger
    print(f'[INFO] Defining Logger(s)...Default -> Tensorboard')
    tb_logger = TensorBoardLogger(
        save_dir=Config.logs_dir,
        name='kaggle-bengali-ai',
        default_hp_metric=False
    )

    # trainer
    print(f'[INFO] Defining Trainer')
    trainer = Trainer(
        gpus=1,
        precision=32,
        # fast_dev_run=True,
        max_epochs=Config.epochs,
        min_epochs=2,
        # plugins = 'deepspeed'
        logger=tb_logger,
        callbacks=callbacks_list
    )
    try:
        # run training job
        print(f'[INFO] Running training job for {Config.epochs} epochs\n\
        train batch size : {Config.train_batch_size}\n\
        test batch size : {Config.test_batch_size}\n\
        lr : {Config.learning_rate}\n\
        Base model : {Config.base_model}'

              )
        trainer, model = train_model(
            model=model,
            datamodule=dm,
            trainer=trainer
        )

        # save model for inference
        print(f'[INFO] Saving model for later inference...')
        th.jit.save(
            model.to_torchscript(),
            os.path.join(Config.models_dir, 'grapheme-classifier-3-in-1.pt')
        )
        print(f'[INFO] Process completed !')
        print(trainer.logged_metrics)
    except Exception as ex:
        print(f'[ERROR] {ex}')
