import torch as th
import torch.nn as nn
from torch.nn import LogSoftmax

# optimization
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, StepLR, ExponentialLR


# vision packages/models packages
from torchvision import models, transforms
import timm  # for some extra architectures such as se_resnet, resnext, efficientnet, etc

import os
import pandas as pd
import numpy as np
import sys
import gc

import pytorch_lightning as pl
from pytorch_lightning import seed_everything
import pytorch_lightning.metrics.functional as metrics

try:
    from .config import Config
except ImportError:
    from config import Config

# seed
seed_everything(Config.seed_val)


class GraphemeClassifier(pl.LightningModule):
    """
        A 3 in 1 model which take the image as input and then:
            - extracts features (encoder)
            - classifies the recognized grapheme root into 168 classes (grapheme_root_decoder)
            - classifies the recognized vowel diacritic into 11 classes (vowel_diacritic_decoder)
            - classifies the recognized consonant diacritic 7 classes (consonant_diacritic_decoder)

        :params :
    """

    def __init__(self,
                 base_encoder: str = 'resnext50_32x4d',
                 arch_from: str = 'timm',
                 vowels_class_weight=None,
                 g_root_class_weight=None,
                 consonant_class_weight=None,
                 dropout=.2,
                 lr=Config.learning_rate,
                 pretrained=False

                 ):
        super(GraphemeClassifier, self).__init__()

        self.save_hyperparameters()

        # defining encoder
        try:
            if self.hparams.arch_from == 'torchvision':
                self.extractor = getattr(
                    models, self.hparams.base_encoder
                )(pretrained=self.hparams.pretrained)
            else:
                self.extractor = timm.create_model(
                    self.hparams.base_encoder,
                    features_only=False,
                    pretrained=self.hparams.pretrained
                )
        except Exception as ex:
            # print(f'[ERROR] {ex}')
            self.extractor = None

        # print(self.extractor)
        self.encoder = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=3,
                kernel_size=(7, 7),
                stride=(2, 2),
                padding=(3, 3),
                bias=False
            ),
            self.extractor
        )

        # dropout layer
        self.dropout = nn.Dropout(p=self.hparams.dropout)

        # defining decoders stack
        try:
            self.grapheme_root_decoder = nn.Linear(
                in_features=self.encoder[1].fc.out_features, out_features=168
            )
            self.vowel_diacritic_decoder = nn.Linear(
                in_features=self.encoder[1].fc.out_features, out_features=11
            )
            self.consonant_diacritic_decoder = nn.Linear(
                in_features=self.encoder[1].fc.out_features, out_features=7
            )
        except AttributeError:
            try:
                self.grapheme_root_decoder = nn.Linear(
                    in_features=self.encoder[1].fc_.out_features, out_features=168
                )
                self.vowel_diacritic_decoder = nn.Linear(
                    in_features=self.encoder[1].fc_.out_features, out_features=11
                )
                self.consonant_diacritic_decoder = nn.Linear(
                    in_features=self.encoder[1].fc_.out_features, out_features=7
                )
            except AttributeError:
                pass

            except:
                self.grapheme_root_decoder = nn.Linear(
                    in_features=self.encoder[1].classifier.out_features, out_features=168
                )
                self.vowel_diacritic_decoder = nn.Linear(
                    in_features=self.encoder[1].classifier.out_features, out_features=11
                )
                self.consonant_diacritic_decoder = nn.Linear(
                    in_features=self.encoder[1].classifier.out_features, out_features=7
                )

    def configure_optimizers(self):

        optimizer = optim.AdamW(
            params=self.parameters(),
            lr=self.hparams.lr,
            eps=1e-8,
            weight_decay=1e-3
        )

        scheduler = None

        return [optimizer], [scheduler]

    def compute_eval_metrics(self, preds: dict, targets: dict):
        """
            Compute evaluation metrics
        """
        g_root_metric = metrics.recall(
            preds=preds['g_logits'],
            target=targets['g_targets'],
            average="macro",
            num_classes=168,
            is_multiclass=True
        )
        vowels_metric = metrics.recall(
            preds=preds['v_logits'],
            target=targets['v_targets'],
            average="macro",
            num_classes=11,
            is_multiclass=True
        )
        consonant_metric = metrics.recall(
            preds=preds['c_logits'],
            target=targets['c_targets'],
            average="macro",
            num_classes=7,
            is_multiclass=True
        )

        eval_metrics = {
            "g_metric": g_root_metric,
            "v_metric": vowels_metric,
            "c_metric": consonant_metric
        }

        return eval_metrics

    def compute_losses(self, logits: dict, targets: dict):
        """
            Compute losses using using crossentropy loss fn
        """
        # grapheme root loss
        g_loss = nn.NLLLoss(
            weight=self.hparams.g_root_class_weight
        )(logits['g_logits'].detach().cpu(), targets['g_targets'].detach().cpu())

        # vowels diacritic loss
        v_loss = nn.NLLLoss(
            weight=self.hparams.vowels_class_weight
        )(logits['v_logits'].detach().cpu(), targets['v_targets'].detach().cpu())

        # consonant diacritic loss
        c_loss = nn.NLLLoss(
            weight=self.hparams.consonant_class_weight
        )(logits['c_logits'].detach().cpu(), targets['c_targets'].detach().cpu())

        # loss map
        losses = {
            "g_loss": g_loss,
            "v_loss": v_loss,
            "c_loss": c_loss
        }
        return losses

    def get_predictions(self, logits: dict):
        """
            convert logits to predictions
        """
        try:
            g_preds = logits['g_logits'].detach().cpu().argmax(dim=1)
            v_preds = logits['v_logits'].detach().cpu().argmax(dim=1)
            c_preds = logits['c_logits'].detach().cpu().argmax(dim=1)

            preds = {
                "g_preds": g_preds,
                "v_preds": v_preds,
                "c_preds": c_preds
            }

            return preds
        except TypeError:
            print('[ERROR] logits are not well given')

    def forward(self, x):
        if self.extractor is not None:
            # extract features
            features = self.encoder(x)

            # apply dropout layer
            if self.hparams.dropout > 0:
                out = self.dropout(features)

            # make classifications
            g = self.grapheme_root_decoder(out)
            v = self.vowel_diacritic_decoder(out)
            c = self.consonant_diacritic_decoder(out)

            logits = {
                "g_logits": LogSoftmax(dim=-1)(g),
                "v_logits": LogSoftmax(dim=-1)(v),
                "c_logits": LogSoftmax(dim=-1)(c)
            }

            return logits
        else:
            print('[ERROR] extractor not found ')
            return None, None, None

    def training_step(self, batch_idx, batch):
        images = batch['image']
        targets = {
            "g_targets": batch['grapheme_root'],
            "v_targets": batch['vowel_diacritic'],
            "c_targets": batch['consonant_diacritic']
        }

        # make prediction
        logits = self(x=images)

        # compute losses
        losses = self.compute_losses(
            logits=logits,
            targets=targets
        )

        # get predictions
        preds = self.get_predictions(logits=logits)

        # compute metrics
        eval_metrics = self.compute_eval_metrics(
            preds=preds,
            targets=targets
        )

    def validation_step(self, batch_idx, batch):
        images, g_targets, v_targets, c_targets = batch
        pass


if __name__ == '__main__':

    dummy_data = th.rand((4, 1, Config.height, Config.width)).cuda()
    dummy_targets = {
        "g_targets": th.randint(low=0, high=168, size=[4]),
        "v_targets": th.randint(low=0, high=11, size=[4]),
        "c_targets": th.randint(low=0, high=7, size=[4])
    }

    print(dummy_targets)
    net = GraphemeClassifier(arch_from="timm").cuda()

    logits = net(dummy_data)
    g_logits = logits["g_logits"]
    v_logits = logits['v_logits']
    c_logits = logits["c_logits"]

    preds = net.get_predictions(logits=logits)
    losses = net.compute_losses(logits=logits, targets=dummy_targets)

    try:
        print(f"g_logits : {g_logits.shape}")
        print(f"v_logits : {v_logits.shape}")
        print(f"c_logits : {c_logits.shape}")

        print(preds)
        print(losses)
    except Exception as ex:
        print(ex)

    gc.collect()
