from config import LIDConfig

from argparse import ArgumentParser
from multiprocessing import Pool
import os
from IPython import embed

from LID.datasetLID import LIDDataset
from LID.lightning_model_l import LightningModel


from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score
import pytorch_lightning as pl


import torch
import torch.utils.data as data

from tqdm import tqdm 
import pandas as pd
import numpy as np

import torch.nn.utils.rnn as rnn_utils
def collate_fn(batch):
    (seq, label) = zip(*batch)
    seql = [x.reshape(-1,) for x in seq]
    seq_length = [x.shape[0] for x in seql]
    data = rnn_utils.pad_sequence(seql, batch_first=True, padding_value=0)
    return data, label, seq_length

if __name__ == "__main__":

    parser = ArgumentParser(add_help=True)
    parser.add_argument('--train_path', type=str, default=LIDConfig.train_path)
    parser.add_argument('--val_path', type=str, default=LIDConfig.val_path)
    parser.add_argument('--test_path', type=str, default=LIDConfig.test_path)
    parser.add_argument('--batch_size', type=int, default=LIDConfig.batch_size)
    parser.add_argument('--epochs', type=int, default=LIDConfig.epochs)
    parser.add_argument('--num_layers', type=int, default=LIDConfig.num_layers)
    parser.add_argument('--feature_dim', type=int, default=LIDConfig.feature_dim)
    parser.add_argument('--lr', type=float, default=LIDConfig.lr)
    parser.add_argument('--gpu', type=int, default=LIDConfig.gpu)
    parser.add_argument('--n_workers', type=int, default=LIDConfig.n_workers)
    parser.add_argument('--dev', type=str, default=False)
    parser.add_argument('--model_checkpoint', type=str, default=LIDConfig.model_checkpoint)
    parser.add_argument('--model_type', type=str, default=LIDConfig.model_type)
    parser.add_argument('--upstream_model', type=str, default=LIDConfig.upstream_model)
    parser.add_argument('--unfreeze_last_conv_layers', action='store_true')
    
    parser = pl.Trainer.add_argparse_args(parser)
    hparams = parser.parse_args()
    print(f'Testing Model on LID Dataset\n#Cores = {hparams.n_workers}\t#GPU = {hparams.gpu}')

    # Testing Dataset
    test_set = LIDDataset(
        wavscp = hparams.test_path,
        hparams = hparams,
        is_train=False
    )
    
    ## Testing Dataloader
    testloader = data.DataLoader(
        test_set, 
        batch_size=1, 
        shuffle=False, 
        num_workers=hparams.n_workers,
        collate_fn = collate_fn,
    )

    #Testing the Model
    if hparams.model_checkpoint:
        model = LightningModel.load_from_checkpoint(hparams.model_checkpoint, HPARAMS=vars(hparams))
        model.eval()

        language_pred = []
        language_true = []

        for batch in tqdm(testloader):
            x, y_l, x_len = batch
            y_l = torch.stack(y_l).reshape(-1,)

            for i in range(x.shape[0]):
                torch.narrow(x, 1, 0, x_len[i])
            y_hat_l = model(x)
                
            language_pred.append(y_hat_l.argmax(dim=1)[0])
            language_true.append(y_l[0])
                
        language_true = np.array(language_true)
        language_pred = np.array(language_pred)
        
        print(accuracy_score(language_true, language_pred))

    else:
        print('Model chekpoint not found for Testing !!!')
