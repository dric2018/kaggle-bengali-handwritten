import os
import pandas as pd
import gc
import matplotlib.pyplot as plt

from .config import Config
from .dataset import DataModule


def get_parquet_lists():
    """
    Load all .parquet files and get train and test splits
    """
    
    parquet_files = [f for f in os.listdir(Config.data_dir) if f.endswith(".parquet")]
    train_files = [f for f in parquet_files if 'train' in f]
    test_files = [f for f in parquet_files if 'test' in f]

    return train_files, test_files



def show_batch(df:pd.DataFrame, subset="valid"):
    
    dm = DataModule(
        df=df, 
        frac=1, 
        validation_split=.2, 
        train_batch_size=Config.train_batch_size, 
        test_batch_size=Config.test_batch_size
    )
    
    # call setup fn 
    dm.setup()
    
    if subset == "valid":
        batch = next(iter(dm.valid_dataloader()))
    else:
        batch = next(iter(dm.train_dataloader()))
        
    images = batch['image']
    g_root = batch['grapheme_root']
    vowel_diacritic = batch['vowel_diacritic']
    consonant_diacritic = batch['consonant_diacritic']

    fig, axes = plt.subplots(ncols=5, nrows=5, figsize=(22, 10), constrained_layout=True)
    i = 0
    for row in range(5):
        for col in range(5):
            axes[row, col].imshow(images[i])
            # Hide grid lines
            #axes[row, col].grid(False)

            # Hide axes ticks
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            
            # set labels
            axes[row, col].set_title(
                f"Grapheme root n°{g_root[i]}\n \
    Vowel diacritic n°{vowel_diacritic[i]}\n \
    consonant diacritic n°{consonant_diacritic[i]}"
            )
            i+=1

    plt.axis('off')
    plt.show()