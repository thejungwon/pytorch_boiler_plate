import os, gc, yaml

import numpy as np
import torch

# logging
import logging
logging.basicConfig(level=logging.DEBUG)
logging.getLogger('matplotlib.font_manager').disabled=True

# warnings
import warnings
warnings.filterwarnings(action='ignore')

# modules
from trainer import Trainer
from model import MyModel
from dataloader import get_loaders_fromfile


if __name__ == '__main__':
    # ===== SET GPU DEVICE =====
    no_cuda = False
    use_cuda = not no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    with open('config.yml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    
    
    DATAPATH = config['data_path']
    TRAIN_DATA = config['train_data']
    VALID_DATA = config['valid_data']
    TEST_DATA = config['test_data'] 
    
    EPOCHS = config['epochs']
    BATCH_SIZE = config['batch_size']
    IN_FEATURES = config['in_features']
    OUT_FEATURES = config['out_features']


    train_loader, valid_loader, test_loader = get_loaders_fromfile(DATAPATH, TRAIN_DATA, VALID_DATA, TEST_DATA, BATCH_SIZE)

    model = MyModel(IN_FEATURES, OUT_FEATURES).to(device)
    optimizer = None
    criterion = None
    scheduler = None
    trainer_class = Trainer(model, optimizer, criterion, scheduler)
    
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(EPOCHS):
        
        avg_train_loss = trainer_class.trainer(BATCH_SIZE, train_loader, scaler, scheduler)
        train_history = np.append(train_history, avg_train_loss.detach().cpu())

        Y_train_lst, Y_train_pred_lst, avg_valid_loss, Y_valid_lst, Y_valid_pred_lst = \
            trainer_class.validate(BATCH_SIZE, train_loader, valid_loader, scaler)
        valid_history = np.append(valid_history, avg_valid_loss.detach().cpu()) 
        
        # ===== OTHER METRIC CALCULATION ======
        
        
        # ===== SCHEDULER AFTER EACH EPOCH =====
        scheduler.step(avg_train_loss)
        
        
        # ===== LOSSS LOG =====
        logging.debug('[Epoch: {:>4}/{}] \ttrain_loss = {:>.9} \tvalid_loss = {:>.9}  \tLR = {}'.format(
            epoch + 1, EPOCHS, avg_train_loss, avg_valid_loss,  scheduler.optimizer.state_dict()['param_groups'][0]['lr']))    