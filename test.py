from config import LIDConfig

from argparse import ArgumentParser
from multiprocessing import Pool
import os

from LID.datasetLID import LIDDataset
from LID.lightning_model_l import LightningModel


from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score
import pytorch_lightning as pl


import torch
import torch.utils.data as data

from tqdm import tqdm 
import pandas as pd
import numpy as np

if __name__ == "__main__":

    parser = ArgumentParser(add_help=True)
    parser.add_argument('--data_path', type=str, default=LIDConfig.data_path)
    parser.add_argument('--speaker_csv_path', type=str, default=LIDConfig.speaker_csv_path)
    parser.add_argument('--timit_wav_len', type=int, default=LIDConfig.timit_wav_len)
    parser.add_argument('--batch_size', type=int, default=LIDConfig.batch_size)
    parser.add_argument('--epochs', type=int, default=LIDConfig.epochs)
    parser.add_argument('--alpha', type=float, default=LIDConfig.alpha)
    parser.add_argument('--beta', type=float, default=LIDConfig.beta)
    parser.add_argument('--gamma', type=float, default=LIDConfig.gamma)
    parser.add_argument('--num_layers', type=int, default=LIDConfig.num_layers)
    parser.add_argument('--feature_dim', type=int, default=LIDConfig.feature_dim)
    parser.add_argument('--lr', type=float, default=LIDConfig.lr)
    parser.add_argument('--gpu', type=int, default=LIDConfig.gpu)
    parser.add_argument('--n_workers', type=int, default=LIDConfig.n_workers)
    parser.add_argument('--dev', type=str, default=False)
    parser.add_argument('--model_checkpoint', type=str, default=LIDConfig.model_checkpoint)
    parser.add_argument('--noise_dataset_path', type=str, default=LIDConfig.noise_dataset_path)
    parser.add_argument('--upstream_model', type=str, default=LIDConfig.upstream_model)
    parser.add_argument('--model_type', type=str, default=LIDConfig.model_type)
    parser.add_argument('--training_type', type=str, default=LIDConfig.training_type)
    parser.add_argument('--data_type', type=str, default=LIDConfig.data_type)
    parser.add_argument('--speed_change', action='store_true')
    parser.add_argument('--unfreeze_last_conv_layers', action='store_true')
    
    parser = pl.Trainer.add_argparse_args(parser)
    hparams = parser.parse_args()
    print(f'Testing Model on NISP Dataset\n#Cores = {hparams.n_workers}\t#GPU = {hparams.gpu}')

    # Testing Dataset
    test_set = LIDDataset(
        wav_folder = os.path.join(hparams.data_path, 'TEST'),
        hparams = hparams,
        is_train=False
    )

    csv_path = hparams.speaker_csv_path
    df = pd.read_csv(csv_path)
    h_mean = df['height'].mean()
    h_std = df['height'].std()
    a_mean = df['age'].mean()
    a_std = df['age'].std()

    #Testing the Model
    if hparams.model_checkpoint:
        if LIDConfig.training_type == 'AHG':
            model = LightningModel.load_from_checkpoint(hparams.model_checkpoint, HPARAMS=vars(hparams))
            model.eval()
            height_pred = []
            height_true = []
            age_pred = []
            age_true = []
            gender_pred = []
            gender_true = []


            # i = 0 
            for batch in tqdm(test_set):
                x, y_h, y_a, y_g = batch
                y_hat_h, y_hat_a, y_hat_g = model(x)

                height_pred.append((y_hat_h*h_std+h_mean).item())
                age_pred.append((y_hat_a*a_std+a_mean).item())
                gender_pred.append(y_hat_g>0.5)

                height_true.append((y_h*h_std+h_mean).item())
                age_true.append(( y_a*a_std+a_mean).item())
                gender_true.append(y_g)

                # if i> 5: break
                # i += 1
            female_idx = np.where(np.array(gender_true) == 1)[0].reshape(-1).tolist()
            male_idx = np.where(np.array(gender_true) == 0)[0].reshape(-1).tolist()

            height_true = np.array(height_true)
            height_pred = np.array(height_pred)
            age_true = np.array(age_true)
            age_pred = np.array(age_pred)


            hmae = mean_absolute_error(height_true[male_idx], height_pred[male_idx])
            hrmse = mean_squared_error(height_true[male_idx], height_pred[male_idx], squared=False)
            amae = mean_absolute_error(age_true[male_idx], age_pred[male_idx])
            armse = mean_squared_error(age_true[male_idx], age_pred[male_idx], squared=False)
            print(hrmse, hmae, armse, amae)

            hmae = mean_absolute_error(height_true[female_idx], height_pred[female_idx])
            hrmse = mean_squared_error(height_true[female_idx], height_pred[female_idx], squared=False)
            amae = mean_absolute_error(age_true[female_idx], age_pred[female_idx])
            armse = mean_squared_error(age_true[female_idx], age_pred[female_idx], squared=False)
            print(hrmse, hmae, armse, amae)
            
            gender_pred_ = [int(pred[0][0] == True) for pred in gender_pred]
            print(accuracy_score(gender_true, gender_pred_))
        
        else:
            model = LightningModel.load_from_checkpoint(hparams.model_checkpoint, HPARAMS=vars(hparams))
            model.eval()
            height_pred = []
            height_true = []
            gender_true = []

            for batch in tqdm(test_set):
                x, y_h, y_a, y_g = batch
                y_hat_h = model(x)

                height_pred.append((y_hat_h*h_std+h_mean).item())
                height_true.append((y_h*h_std+h_mean).item())
                gender_true.append(y_g)

            female_idx = np.where(np.array(gender_true) == 1)[0].reshape(-1).tolist()
            male_idx = np.where(np.array(gender_true) == 0)[0].reshape(-1).tolist()

            height_true = np.array(height_true)
            height_pred = np.array(height_pred)

            hmae = mean_absolute_error(height_true[male_idx], height_pred[male_idx])
            hrmse = mean_squared_error(height_true[male_idx], height_pred[male_idx], squared=False)
            print(hrmse, hmae)

            hmae = mean_absolute_error(height_true[female_idx], height_pred[female_idx])
            hrmse = mean_squared_error(height_true[female_idx], height_pred[female_idx], squared=False)
            print(hrmse, hmae)


    else:
        print('Model chekpoint not found for Testing !!!')
