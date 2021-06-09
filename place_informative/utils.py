from scipy.sparse.construct import random
from sklearn.utils.validation import _deprecate_positional_args
from torch.utils import data
from torchvision import transforms as T
from pathlib import Path
import glob
from scipy.io import loadmat
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from hydra.utils import to_absolute_path
import hydra
import random
import json


import torch
import torch.nn.functional as F
import torch.nn as nn


# imgae transforms
class transforms:
    """
    data transform e.g. data augmentation or to tensor transform
    All Images will be resize and transform to torch.tensor
    

    Parameters
    ----------
    config parameters:
        img_size: int
            image size to resize the input images

    Returns
    ----------
    Transform object

    """
    def __init__(self, cfg):
        self.img_size = cfg.model.img_size

    def get_transform(self, mode):
        transform = []
        
        if mode == 'train':
            transform.append(T.RandomResizedCrop(self.img_size))
            transform.append(T.RandomHorizontalFlip())

        elif mode == 'valid' or mode == 'test':
            transform.append(T.Resize(256))
            transform.append(T.CenterCrop(self.img_size))
            
        else:
            raise ValueError("mode while calling transform when specifying dataloader shall be selected properly e.g: 'train', 'valid' or 'test ")

        transform.append(T.ToTensor())
        transform.append(T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))

        return T.Compose(transform)

# train, val & test split
class data_split:
    def __init__(self, cfg, mode):
        self.mode = mode
        self.test_split = cfg.data.test_split
        self.val_split = cfg.data.val_split
        self.root = cfg.data.root
        self.class_map_path = f'{self.root}/class_map.json'
        
    def split(self):
        root = Path(self.root)
        imgs_path = glob.glob(f'{root}/**/*.jpg')
        train_val_path, test_path = train_test_split(imgs_path, test_size=self.test_split, shuffle=True)
        train_path, val_path = train_test_split(train_val_path, test_size=self.val_split, shuffle=True)
        if self.mode == 'train':
            data_set = train_path
        elif self.mode == 'val' or self.mode == 'validation' or self.mode == 'valid':
            data_set = val_path
        elif self.mode == 'test':
            data_set = test_path
        else:
            raise ValueError("dataset mode shall be selected properly e.g: 'train', 'valid' or 'validation', 'test ")

        return data_set

    def load_class_map(self):

        with open(self.class_map_path) as json_file:
            class_map = json.load(json_file)

        return class_map


# epoch train and eval
def train_one_epoch(data_loader, model, criterion, optimizer, epoch, device, phase):
    """
    One epoch training code as well as evaluation 


    Parameters
    ----------
    data_loader: torch Dataloader object

    model: torchvison model object
        model made by factory.py
    
    criterion: loss object
        loss function for 'classification' model.

    optimizer: torch optim object
        optimizer to update the parameters in backprob process

    epoch: int 
        epoch no

    device: torch.device
        'cuda' or 'cpu

    phase: string
        'train', 'valid' or 'test'


    Returns
    ----------
    epoch_loss: float
        loss value of the epoch
    
    epoch_acc: float
        accuracy value of the epoch

    """
    
    # Phase selection to compute model output
    if phase == 'train':
        # switch to train mode
        model.train()

    else:
        # switch to evaluate/test mode
        if phase == 'test':
            print('\ntesting the model...')
        else:
            print('\nevaluating the model...')
        model.eval()
    
    
    running_loss = 0.           # running loss is accumulative of each batch loss value
    running_corrects = 0        
    running_count = 0

    ###########################################
    # Classification Model Training
    ###########################################
    
    for inputs, targets in tqdm(data_loader, unit="batch"):
        inputs, targets = inputs.to(device), targets.to(device)

        # compute output
        optimizer.zero_grad()
        with torch.set_grad_enabled(phase=='train'):
            logits, probas = model(inputs)
            _, preds = torch.max(logits, 1)

            # loss calculation
            loss = criterion(logits, targets)

            if phase == 'train':
                # backrpop
                loss.backward()
                # update parameters
                optimizer.step()

        running_loss += loss.item() * targets.size(0)
        batch_corrects = torch.sum(preds == targets.data)
        running_corrects += batch_corrects
        running_count += targets.size(0)
    
    # epoch loss, accuracy, mae and mse
    epoch_loss = running_loss / running_count
    epoch_acc = running_corrects.double() / running_count

    ###########################################
    # Epoch training output
    ###########################################    
    print(f'{phase} loss: {epoch_loss:.4f}          {phase} acc: {epoch_acc:.4f}')

    return (epoch_loss, epoch_acc)


## TO DO
# 
# 


# random.seed(1221)
# np.random.seed(1221)
# torch.manual_seed(1221)
# torch.cuda.manual_seed_all(1221)

# config_name = str(Path(to_absolute_path(__file__)).resolve().parents[0].joinpath('src', 'models', 'config', 'config.yaml'))
# @hydra.main(config_name=config_name)
# def main(cfg):
#     data = data_split(cfg, 'val')
#     class_map = data.load_class_map()
#     print(class_map)

# if __name__ == '__main__':
#     main()
