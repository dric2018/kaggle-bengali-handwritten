import os
import pandas as pd
import gc
import matplotlib.pyplot as plt
import sklearn.metrics
import numpy as np


from .config import Config
from .dataset import DataModule


def get_parquet_lists():
    """
    Load all .parquet files and get train and test splits
    """

    parquet_files = [f for f in os.listdir(
        Config.data_dir) if f.endswith(".parquet")]
    train_files = [f for f in parquet_files if 'train' in f]
    test_files = [f for f in parquet_files if 'test' in f]

    return train_files, test_files


def show_batch(df: pd.DataFrame, subset="valid"):

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

    fig, axes = plt.subplots(ncols=5, nrows=5, figsize=(
        22, 10), constrained_layout=True)
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
            i += 1

    plt.axis('off')
    plt.show()

    return fig


def compute_model_score(solution: pd.DataFrame, submission: pd.DataFrame):
    """"
        from https://www.kaggle.com/c/bengaliai-cv19/overview/evaluation
    """

    scores = []

    for component in ['grapheme_root', 'consonant_diacritic', 'vowel_diacritic']:
        y_true_subset = solution[solution[component]
                                 == component]['target'].values

        y_pred_subset = submission[submission[component]
                                   == component]['target'].values
        scores.append(
            sklearn.metrics.recall_score(
                y_true_subset, y_pred_subset, average='macro'
            )
        )

    final_score = np.average(scores, weights=[2, 1, 1])

    return final_score


def train_model():
    pass


def evaluate_model():
    pass


def run_kFold():
    pass


def run_inference():
    pass
