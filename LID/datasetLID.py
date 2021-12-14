from torch.utils.data import Dataset
import os
import pandas as pd
import torch
import numpy as np

import torchaudio
import wavencoder

from IPython import embed


class LIDDataset(Dataset):
    def __init__(self,
    wavscp,
    hparams,
    is_train=True,
    ):
        self.wavscp = wavscp

        with open(wavscp, 'r') as f:
            lines = f.readlines()
            self.label_list = [i.split()[0][:5] for i in lines]
            self.files = [i.split()[-1] for i in lines]

        self.is_train = is_train
        self.classes = {'Kazak': 0, 'Tibet': 1, 'Uyghu': 2, 'ct-cn': 3, 'id-id': 4, 'ja-jp': 5, 'ko-kr': 6, 'ru-ru': 7, 'vi-vn': 8, 'zh-cn': 9}

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        file = self.files[idx]
        
        language = self.classes[self.label_list[idx]]

        wav, _ = torchaudio.load(file)

        if(wav.shape[0] != 1):
            wav = torch.mean(wav, dim=0)
        
        return wav, torch.LongTensor([language])
