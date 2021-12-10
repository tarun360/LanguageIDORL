from config import TIMITConfig
from argparse import ArgumentParser
from multiprocessing import Pool
import os

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning import Trainer


import torch
import torch.utils.data as data
# torch.use_deterministic_algorithms(True)


# SEED
SEED=100
pl.utilities.seed.seed_everything(SEED)
torch.manual_seed(SEED)


from TIMIT.datasetLID import LIDDataset
from TIMIT.lightning_model_l import LightningModel

if __name__ == "__main__":

    parser = ArgumentParser(add_help=True)
    parser.add_argument('--train_path', type=str, default=TIMITConfig.train_path)
    parser.add_argument('--val_path', type=str, default=TIMITConfig.val_path)
    parser.add_argument('--test_path', type=str, default=TIMITConfig.test_path)
    parser.add_argument('--speaker_csv_path', type=str, default=TIMITConfig.speaker_csv_path)
    parser.add_argument('--timit_wav_len', type=int, default=TIMITConfig.timit_wav_len)
    parser.add_argument('--batch_size', type=int, default=TIMITConfig.batch_size)
    parser.add_argument('--epochs', type=int, default=TIMITConfig.epochs)
    parser.add_argument('--alpha', type=float, default=TIMITConfig.alpha)
    parser.add_argument('--beta', type=float, default=TIMITConfig.beta)
    parser.add_argument('--gamma', type=float, default=TIMITConfig.gamma)
    parser.add_argument('--num_layers', type=int, default=TIMITConfig.num_layers)
    parser.add_argument('--feature_dim', type=int, default=TIMITConfig.feature_dim)
    parser.add_argument('--lr', type=float, default=TIMITConfig.lr)
    parser.add_argument('--gpu', type=int, default=TIMITConfig.gpu)
    parser.add_argument('--n_workers', type=int, default=TIMITConfig.n_workers)
    parser.add_argument('--dev', type=str, default=False)
    parser.add_argument('--model_checkpoint', type=str, default=TIMITConfig.model_checkpoint)
    #     parser.add_argument('--noise_dataset_path', type=str, default=TIMITConfig.noise_dataset_path)
    parser.add_argument('--noise_dataset_path', type=str, default=None)
    parser.add_argument('--model_type', type=str, default=TIMITConfig.model_type)
    parser.add_argument('--upstream_model', type=str, default=TIMITConfig.upstream_model)
    parser.add_argument('--training_type', type=str, default=TIMITConfig.training_type)
    parser.add_argument('--data_type', type=str, default=TIMITConfig.data_type)
    parser.add_argument('--speed_change', action='store_true')
    parser.add_argument('--unfreeze_last_conv_layers', action='store_true')

    parser = pl.Trainer.add_argparse_args(parser)
    hparams = parser.parse_args()
    print(f'Training Model on TIMIT Dataset\n#Cores = {hparams.n_workers}\t#GPU = {hparams.gpu}')

    # Training, Validation and Testing Dataset
    ## Training Dataset
    train_set = LIDDataset(
        wavscp = hparams.train_path,
        hparams = hparams
    )
    ## Training DataLoader
    trainloader = data.DataLoader(
        train_set, 
        batch_size=hparams.batch_size, 
        shuffle=True, 
        num_workers=hparams.n_workers
    )
    ## Validation Dataset
    valid_set = LIDDataset(
        wav_folder = hparams.val_path,
        hparams = hparams,
        is_train=False
    )
    ## Validation Dataloader
    valloader = data.DataLoader(
        valid_set, 
        batch_size=1,
        # hparams.batch_size, 
        shuffle=False, 
        num_workers=hparams.n_workers
    )
    ## Testing Dataset
    test_set = LIDDataset(
        wav_folder = hparams.test_path,
        hparams = hparams,
        is_train=False
    )
    ## Testing Dataloader
    testloader = data.DataLoader(
        test_set, 
        batch_size=1,
        # hparams.batch_size, 
        shuffle=False, 
        num_workers=hparams.n_workers
    )

    print('Dataset Split (Train, Validation, Test)=', len(train_set), len(valid_set), len(test_set))


    # Training the Model
    # logger = TensorBoardLogger('TIMIT_logs', name='')
    logger = WandbLogger(
        name=TIMITConfig.run_name,
        project='SpeakerProfiling'
    )

    model = LightningModel(vars(hparams))

    model_checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints',
        monitor='val/loss', 
        mode='min',
        verbose=1)

    trainer = Trainer(
        fast_dev_run=hparams.dev, 
        gpus=hparams.gpu, 
        max_epochs=hparams.epochs, 
        checkpoint_callback=True,
        callbacks=[
            EarlyStopping(
                monitor='val/loss',
                min_delta=0.00,
                patience=20,
                verbose=True,
                mode='min'
                ),
            model_checkpoint_callback
        ],
        logger=logger,
        resume_from_checkpoint=hparams.model_checkpoint,
        distributed_backend='ddp'
        )

    trainer.fit(model, train_dataloader=trainloader, val_dataloaders=valloader)

    print('\n\nCompleted Training...\nTesting the model with checkpoint -', model_checkpoint_callback.best_model_path)
    #model = LightningModel.load_from_checkpoint(model_checkpoint_callback.best_model_path)
    #trainer.test(model, test_dataloaders=testloader)
