from trainer import Trainer
from model import MyModel
import os, gc, yaml


import torch





if __name__ == '__main__':
    # ===== SET GPU DEVICE =====
    no_cuda = False
    use_cuda = not no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    with open('config.yml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    IN_FEATURES = config['in_features']
    OUT_FEATURES = config['out_features']


    model = MyModel(IN_FEATURES, OUT_FEATURES).to(device)
    optimizer = None
    criterion = None
    scheduler = None
    trainer_class = Trainer(model, optimizer, criterion, scheduler)
