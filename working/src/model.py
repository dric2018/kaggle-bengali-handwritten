import os
from utils import ramp_scheduler
import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import recall

from config import Config
import timm  # for some extra architectures such as se_resnet, resnext, efficientnet, etc

import torch as th
import torch.nn as nn
import torch.nn.functional as F

from torch import optim


class Model(pl.LightningModule):

    # Model architecture
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
        super(Model, self).__init__()

        self.save_hyperparameters()

        # defining encoder
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
        opt = optim.AdamW(
            params=self.parameters(),
            lr=self.hparams.lr,
            eps=1e-8,
            weight_decay=1e-3
        )

        scheduler = th.optim.lr_scheduler.LambdaLR(
            optimizer=opt,
            lr_lambda=ramp_scheduler,
            verbose=True
        )

        return [opt], [scheduler]

    def forward(self, x):
        # extract features
        features = self.encoder(x)

        # apply dropout layer
        out = self.dropout_layer(features)

        # make classifications
        g = self.grapheme_root_decoder(out)
        v = self.vowel_diacritic_decoder(out)
        c = self.consonant_diacritic_decoder(out)

        g_logits = F.log_softmax(g, dim=1)
        v_logits = F.log_softmax(v, dim=1)
        c_logits = F.log_softmax(c, dim=1)

        return (g_logits, v_logits, c_logits)

    def compute_eval_metrics(self, preds: tuple, targets: dict):
        """
            Compute evaluation metrics
        """
        g_preds = preds[0]
        v_preds = preds[1]
        c_preds = preds[2]

        g_root_metric = recall(
            preds=g_preds,
            target=targets['g_targets'],
            average="macro",
            num_classes=168,
            is_multiclass=True
        )
        vowels_metric = recall(
            preds=v_preds,
            target=targets['v_targets'],
            average="macro",
            num_classes=11,
            is_multiclass=True
        )
        consonant_metric = recall(
            preds=c_preds,
            target=targets['c_targets'],
            average="macro",
            num_classes=7,
            is_multiclass=True
        )

        return (g_root_metric, vowels_metric, consonant_metric)

    def compute_avg_recall(self, metrics: tuple):
        g_metric, v_metric, c_metric = metrics[0], metrics[1], metrics[2]
        avg_recall = (g_metric*.5 + v_metric*.25 + c_metric*.25) / 3

        return avg_recall

    def compute_loss(self, logits: tuple, targets: dict):
        """
            Compute losses using using crossentropy loss fn
        """
        g_logits, v_logits, c_logits = logits[0], logits[1], logits[2]

        # grapheme root loss
        g_loss = F.nll_loss(
            weight=self.hparams.g_root_class_weight,
            input=g_logits,
            target=targets['g_targets']
        )

        # vowels diacritic loss
        v_loss = F.nll_loss(
            weight=self.hparams.vowels_class_weight,
            input=v_logits,
            target=targets['v_targets']
        )

        # consonant diacritic loss
        c_loss = F.nll_loss(
            weight=self.hparams.consonant_class_weight,
            input=c_logits,
            target=targets['c_targets']
        )

        return (g_loss*.5 + v_loss*.25 + c_loss*.25) / 3

    def get_predictions(self, logits: tuple):
        """
            convert logits to predictions
        """
        g_logits, v_logits, c_logits = logits[0], logits[1], logits[2]
        g_preds = g_logits.argmax(dim=1)
        v_preds = v_logits.argmax(dim=1)
        c_preds = c_logits.argmax(dim=1)

        return (g_preds, v_preds, c_preds)

    def training_step(self, batch, batch_idx):
        images = batch['image']
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
        images = batch['image']
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
    model = Model(
        base_encoder=Config.base_model,
        arch_from='timm',
        vowels_class_weight=None,
        g_root_class_weight=None,
        consonant_class_weight=None,
        drop=0.3,
        lr=Config.learning_rate,
        pretrained=True
    )

    print(model)
