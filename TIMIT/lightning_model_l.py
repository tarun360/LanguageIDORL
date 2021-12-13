import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl
from pytorch_lightning.metrics.regression import MeanAbsoluteError as MAE
from pytorch_lightning.metrics.regression import MeanSquaredError  as MSE
from pytorch_lightning.metrics.classification import Accuracy

import pandas as pd
import torch_optimizer as optim


from Model.models import UpstreamTransformer
from Model.utils import RMSELoss

from IPython import embed

class LightningModel(pl.LightningModule):
    def __init__(self, HPARAMS):
        super().__init__()
        # HPARAMS
        self.save_hyperparameters()
        self.models = {
            'UpstreamTransformer': UpstreamTransformer
        }
        self.model = self.models[HPARAMS['model_type']](upstream_model=HPARAMS['upstream_model'], num_layers=HPARAMS['num_layers'], feature_dim=HPARAMS['feature_dim'], unfreeze_last_conv_layers=HPARAMS['unfreeze_last_conv_layers'])
        self.classification_criterion = nn.CrossEntropyLoss()
        self.accuracy = Accuracy()
        self.lr = HPARAMS['lr']

        print(f"Model Details: #Params = {self.count_total_parameters()}\t#Trainable Params = {self.count_trainable_parameters()}")

    def count_total_parameters(self):
            return sum(p.numel() for p in self.parameters())
    
    def count_trainable_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return [optimizer]

    def training_step(self, batch, batch_idx):
        x, y_l = batch
        
#         embed()
        
        y_hat_l = self(x)
        
        language_loss = self.classification_criterion(y_hat_l, y_l)

        winners = y_hat_l.argmax(dim=1)
        corrects = (winners == y_l)
        language_acc = corrects.sum().float() / float( y_hat_l.size(0) )

        loss = language_loss

        return {'loss':loss, 
                'language_acc':language_acc,
                }
    
    def training_epoch_end(self, outputs):
        n_batch = len(outputs)
        loss = torch.tensor([x['loss'] for x in outputs]).mean()
        language_acc = torch.tensor([x['language_acc'] for x in outputs]).mean()

        self.log('train/loss' , loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train/acc',language_acc, on_step=False, on_epoch=True, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        x, y_l = batch
        y_hat_l = self(x)
        
        language_loss = self.classification_criterion(y_hat_l, y_l)

        winners = y_hat_l.argmax(dim=1)
        corrects = (winners == y_l)
        language_acc = corrects.sum().float() / float( y_hat_l.size(0) )

        loss = language_loss

        return {'val_loss':loss, 
                'val_language_acc':language_acc,
                }

    def validation_epoch_end(self, outputs):
        val_loss = torch.tensor([x['val_loss'] for x in outputs]).mean()
        language_acc = torch.tensor([x['val_language_acc'] for x in outputs]).mean()
        
        self.log('val/loss' , val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/acc',language_acc, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y_l = batch
        y_hat_l = self(x)

        winners = y_hat_l.argmax(dim=1)
        corrects = (winners == y_l)
        language_acc = corrects.sum().float() / float( y_hat_l.size(0) )

        return {'language_acc':language_acc}

    def test_epoch_end(self, outputs):
        language_acc = torch.tensor([x['language_acc'] for x in outputs]).mean()

        pbar = {'language_acc' : language_acc}
        self.logger.log_hyperparams(pbar)
        self.log_dict(pbar)