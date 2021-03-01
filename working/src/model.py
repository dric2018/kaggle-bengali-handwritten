import torch as th
import torch.nn as nn
import torch.nn.functional as F
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
                 drop=.2,
                 lr=Config.learning_rate,
                 pretrained=False

                 ):
        super().__init__()

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
        self.dropout_layer = nn.Dropout(p=self.hparams.drop)

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
                self.grapheme_root_decoder = nn.Linear(
                    in_features=self.encoder[1].classifier.out_features, out_features=168
                )
                self.vowel_diacritic_decoder = nn.Linear(
                    in_features=self.encoder[1].classifier.out_features, out_features=11
                )
                self.consonant_diacritic_decoder = nn.Linear(
                    in_features=self.encoder[1].classifier.out_features, out_features=7
                )

            except:
                pass

    def configure_optimizers(self):
        optimizer = optim.SGD(
            params=self.parameters(),
            lr=self.hparams.lr,
            # eps=1e-8,
            # weight_decay=1e-3
        )

        scheduler = CosineAnnealingLR(
            optimizer=optimizer,
            T_max=Config.epochs // 5,
            eta_min=Config.learning_rate,
            last_epoch=-1,
            verbose=False
        )

        return [optimizer], [scheduler]

    def compute_eval_metrics(self, preds: dict, targets: dict):
        """
            Compute evaluation metrics
        """
        g_root_metric = metrics.recall(
            preds=preds['g_preds'].cpu(),
            target=targets['g_targets'].cpu(),
            average="macro",
            num_classes=168,
            is_multiclass=True
        )
        vowels_metric = metrics.recall(
            preds=preds['v_preds'].cpu(),
            target=targets['v_targets'].cpu(),
            average="macro",
            num_classes=11,
            is_multiclass=True
        )
        consonant_metric = metrics.recall(
            preds=preds['c_preds'].cpu(),
            target=targets['c_targets'].cpu(),
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

    def compute_avg_recall(self, metrics: dict):

        avg_recall = (metrics['g_metric']*.5 +
                      metrics['v_metric']*.25 + metrics['c_metric']*.5) / 3

        return avg_recall

    def compute_loss(self, logits: dict, targets: dict):
        """
            Compute losses using using crossentropy loss fn
        """
        # grapheme root loss
        g_loss = F.nll_loss(
            weight=self.hparams.g_root_class_weight,
            input=logits['g_logits'].cpu(),
            target=targets['g_targets'].cpu()
        )

        # vowels diacritic loss
        v_loss = F.nll_loss(
            weight=self.hparams.vowels_class_weight,
            input=logits['v_logits'].cpu(),
            target=targets['v_targets'].cpu()
        )

        # consonant diacritic loss
        c_loss = F.nll_loss(
            weight=self.hparams.consonant_class_weight,
            input=logits['c_logits'].cpu(),
            target=targets['c_targets'].cpu()
        )

        return (g_loss*.5 + v_loss*.25 + c_loss*.25) / 3

    def get_predictions(self, logits: dict):
        """
            convert logits to predictions
        """
        try:
            g_preds = logits['g_logits'].argmax(dim=1)
            v_preds = logits['v_logits'].argmax(dim=1)
            c_preds = logits['c_logits'].argmax(dim=1)

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
            out = self.dropout_layer(features)

            # make classifications
            g = self.grapheme_root_decoder(out)
            v = self.vowel_diacritic_decoder(out)
            c = self.consonant_diacritic_decoder(out)

            logits = {
                "g_logits": F.log_softmax(g, dim=1),
                "v_logits": F.log_softmax(v, dim=1),
                "c_logits": F.log_softmax(c, dim=1)
            }

            return logits
        else:
            print('[ERROR] extractor not found ')
            return None, None, None

    def training_step(self, batch, batch_idx):
        images = batch['image'].unsqueeze(1)
        targets = {
            "g_targets": batch['grapheme_root'],
            "v_targets": batch['vowel_diacritic'],
            "c_targets": batch['consonant_diacritic']
        }

        # make prediction
        logits = self(x=images)

        # get predictions
        preds = self.get_predictions(logits=logits)

        # compute metrics
        eval_metrics = self.compute_eval_metrics(
            preds=preds,
            targets=targets
        )

        # model score
        train_recall = self.compute_avg_recall(metrics=eval_metrics)
        # loss
        train_loss = self.compute_loss(
            logits=logits,
            targets=targets
        )
        # log metrics

        self.log(
            'train_loss',
            train_loss,
            prog_bar=True,
            logger=True,
            on_epoch=True,
            on_step=True
        )
        self.log(
            'train_recall',
            train_recall,
            prog_bar=True,
            logger=True,
            on_epoch=True,
            on_step=True
        )

        return {
            'loss': train_loss,
            'recall': train_recall,
            "predictions": preds,
            'targets': targets
        }

    def training_epoch_end(self, outputs):
        #  the function is called after every epoch is completed

        # calculating average losses
        avg_loss = th.stack([x['loss'] for x in outputs]).mean()

        # recall
        avg_recall = th.stack([x['recall'] for x in outputs]).mean()

        # logging using tensorboard logger
        self.logger.experiment.add_scalar("Loss/Train",
                                          avg_loss,
                                          self.current_epoch
                                          )

        self.logger.experiment.add_scalar("Recall/Train",
                                          avg_recall,
                                          self.current_epoch
                                          )

    def validation_step(self, batch, batch_idx):
        images = batch['image'].unsqueeze(1)
        targets = {
            "g_targets": batch['grapheme_root'],
            "v_targets": batch['vowel_diacritic'],
            "c_targets": batch['consonant_diacritic']
        }

        # make prediction
        logits = self(x=images)

        # get predictions
        preds = self.get_predictions(logits=logits)

        # compute metrics
        eval_metrics = self.compute_eval_metrics(
            preds=preds,
            targets=targets
        )

        # model score
        val_recall = self.compute_avg_recall(metrics=eval_metrics)
        # loss
        val_loss = self.compute_loss(
            logits=logits,
            targets=targets
        )

        # log metrics
        self.log(
            "val_recall",
            val_recall,
            prog_bar=True,
            logger=True,
            on_epoch=True,
            on_step=False
        )

        self.log(
            "val_loss",
            val_loss,
            prog_bar=True,
            logger=True,
            on_epoch=True,
            on_step=False
        )

        return {
            'loss': val_loss,
            'recall': val_recall,
            "predictions": preds,
            'targets': targets
        }

    def validation_epoch_end(self, outputs):
        #  the function is called after every epoch is completed

        # calculating average losses
        avg_loss = th.stack([x['loss'] for x in outputs]).mean()

        # recall
        avg_recall = th.stack([x['recall'] for x in outputs]).mean()

        # logging using tensorboard logger
        self.logger.experiment.add_scalar("Loss/Validation",
                                          avg_loss,
                                          self.current_epoch
                                          )

        self.logger.experiment.add_scalar("Recall/Validation",
                                          avg_recall,
                                          self.current_epoch
                                          )


if __name__ == '__main__':

    batch_size = 8

    dummy_data = th.rand((batch_size, 1, Config.height, Config.width)).cuda()
    dummy_targets = {
        "g_targets": th.randint(low=0, high=168, size=[batch_size]),
        "v_targets": th.randint(low=0, high=11, size=[batch_size]),
        "c_targets": th.randint(low=0, high=7, size=[batch_size])
    }

    print(dummy_targets)
    net = GraphemeClassifier(
        arch_from="timm", base_encoder="densenet169").cuda()

    try:
        logits = net(dummy_data)
        g_logits = logits["g_logits"]
        v_logits = logits['v_logits']
        c_logits = logits["c_logits"]

        preds = net.get_predictions(logits=logits)
        loss = net.compute_loss(logits=logits, targets=dummy_targets)

        print(f"g_logits : {g_logits.shape}")
        print(f"v_logits : {v_logits.shape}")
        print(f"c_logits : {c_logits.shape}")

        print(preds)
        print(th.tensor(list(losses.values())).mean())

    except Exception as ex:
        print(ex)

    gc.collect()
