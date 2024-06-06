import json
import cv2
import numpy as np
import pandas as pd
import pickle

from torch.utils.data import Dataset


class MIPDataset(Dataset):
    def __init__(self, dataset_path):
        df = pd.read_excel(dataset_path)
        df = df.loc[df['allocated_set']=='train_val'].reset_index(drop=True)
        self.df = df

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        row_idx = idx

        source_path = self.df.loc[row_idx, 'SEG_MIP_path']  #SEG_MIP
        target_path = self.df.loc[row_idx, 'SUV_MIP_path']  #SUV_MIP
        prompt = self.df.loc[row_idx, 'prompt']
        
        with open(source_path, 'rb') as f:
            source = pickle.load(f)
        with open(target_path, 'rb') as f:
            target = pickle.load(f)
        
        height,width = source.shape
        source = np.broadcast_to(source[:,:,np.newaxis], (height,width,3))
        target = np.broadcast_to(target[:,:,np.newaxis], (height,width,3))
        
        source = source.astype(np.float32)  #range [0,1]
        
        # Normalize target images to [-1, 1].
        # Min SUV value is 0. Max SUV value is around 500.
        target = (target.astype(np.float32) / 250.0) - 1.0
        #default code for general images having pixel values in the range [0,255]
        #target = (target.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=target, txt=prompt, hint=source)

