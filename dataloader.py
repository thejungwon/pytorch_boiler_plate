import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms


class CustomDataset(Dataset):
    def __init__(self, data_path):
        
        pass
        
            
def get_loaders_fromfile(datapath, train_data, valid_data, test_data, batch_size):
    
    #-- Load train data with batch
    train_loader = DataLoader(dataset=CustomDataset(f'{datapath}{train_data}')
                              , batch_size=batch_size
                              , shuffle=True
                              , drop_last=True
                              , num_workers=4
                              , pin_memory=True
                             )
    
    #-- Load valid data with batch
    valid_loader = DataLoader(dataset=CustomDataset(f'{datapath}{valid_data}')
                              , batch_size=batch_size
                              , shuffle=False
                              , drop_last=True
                              , num_workers=4
                              , pin_memory=True
                             )
    
    #-- Load Test data with batch
    test_loader = DataLoader(dataset=CustomDataset(f'{datapath}{test_data}')
                             , batch_size=batch_size
                             , shuffle=False
                             , drop_last=True
                             , num_workers=4
                             , pin_memory=True
                            )

    return train_loader, valid_loader, test_loader


