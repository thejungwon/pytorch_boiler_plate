from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import logging


# ===== SET GPU DEVICE =====
no_cuda = False
use_cuda = not no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


# ===== TRAINING =====
class Trainer():
    def __init__(self, model, optimizer, criterion, scheduler):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        

        
    def trainer(self, batch_size, train_loader, scaler):
        # Set to train mode
        self.model.train()
        
        avg_loss_train = 0   #epoch 단위 train loss
        for batch_idx, (data, targets) in enumerate(train_loader):
            #-- Get data to cuda if possible
            X_train, Y_train = data.to(device=device, dtype=torch.float32), targets.to(device=device, dtype=torch.uint8)
            
            #-- Clear gradients w.r.t. parameters
            self.optimizer.zero_grad()

            #-- calculating loss prediction
            with torch.cuda.amp.autocast():
                train_pred = self.model(X_train)

            #-- Calculating Cost
            train_loss = self.criterion(train_pred, Y_train)
            
            
            #-- Getting gradients w.r.t. parameters
            scaler.scale(train_loss).backward()

            #-- Cost 개선
            scaler.step(self.optimizer)
                   
            #-- updates the scale for next iter
            scaler.update()
            
            #-- scheduler step
            self.scheduler.step()

            avg_loss_train += train_loss / len(train_loader) #loss/total batch
        return avg_loss_train

    
    def validate(self, batch_size, train_loader, valid_loader, scaler):
        # Set to evaluation mode 
        self.model.eval()
        
        Y_train_lst = np.array([])
        Y_train_pred_lst = np.array([])
        Y_valid_lst = np.array([])
        Y_valid_pred_lst = np.array([])
        
        # Turn on the no_grad mode to make more efficient
        with torch.no_grad():
            avg_loss_valid = 0
            
            for batch_idx, (data, targets) in enumerate(train_loader):
                # Get data to cuda if possible
                X_train = data.to(device, dtype=torch.float32)
                Y_train_lst = np.append(Y_train_lst, targets.detach().cpu()) 
                Y_train = targets.to(device, dtype=torch.uint8)

                #== Calculating Loss
                train_pred = self.model(X_train)
                Y_train_pred_lst = np.append(Y_train_pred_lst, train_pred.detach().cpu())
                    
            for batch_idx, (data, targets) in enumerate(valid_loader):
                # Get data to cuda if possible
                X_valid = data.to(device, dtype=torch.float32)
                Y_valid_lst = np.append(Y_valid_lst, targets.detach().cpu()) 
                Y_valid = targets.to(device, dtype=torch.uint8)

                #== Calculating Loss
                valid_pred = self.model(X_valid)

                Y_valid_pred_lst = np.append(Y_valid_pred_lst, valid_pred.detach().cpu())

                #== Calculating Cost
                valid_loss = self.criterion(valid_pred, Y_valid)  # batch loss
                
                
                avg_loss_valid += valid_loss / len(valid_loader) #loss/total batch

        return Y_train_lst, Y_train_pred_lst, avg_loss_valid, Y_valid_lst, Y_valid_pred_lst
