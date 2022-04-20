import hydra
from pathlib import Path
from hydra.utils import to_absolute_path
import numpy as np
import random
from tqdm import tqdm
import datetime
from sacred import Experiment
from sacred.observers import MongoObserver

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter

from place_informative.src.models.factory import ModelBuild
from place_informative.data.generator import DatasetLoader
from place_informative.utils import transforms, train_one_epoch


# config file path
config_name = str(Path(to_absolute_path(__file__)).resolve().parents[1].joinpath('models', 'config', 'config.yaml'))

# Device 
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print (f'\ntraining on {device} . . .')

ex = Experiment('urban_train')
ex.add_config(config_name)
ex.observers.append(MongoObserver(db_name='urban'))

@ex.capture
def train(_config, _run):
    # Random seed
    if _config['seed']:
        seed=_config['seed']
        random.seed(seed) 
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)                                         
    ## Data loading code
    T = transforms(_config)
    train_loader = DatasetLoader(_config, 'train', transforms=T.get_transform('train')).Loader() 
    val_loader = DatasetLoader(_config, 'valid', transforms=T.get_transform('valid')).Loader()
    test_loader = DatasetLoader(_config, 'test', transforms=T.get_transform('test')).Loader()

    
    # create model
    model = ModelBuild(_config)

    # SummaryWriter
    if _config['log_records'] == True:
        log_dir = Path(to_absolute_path(__file__)).resolve().parents[2].joinpath('reports', 'tb_log', _config['model_name'], 
                                                                                 datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
        writer = SummaryWriter(log_dir=log_dir)
        # writer.add_graph(model, torch.rand((cfg.train.batch_size, 3, cfg.model.img_size, cfg.model.img_size)))  # uncomment these lines in
        # writer.flush()                                                                                          # to have model graph in tensorboard
    
    # sending the model to device and calculate total and trainable parameters
    model.to(device)
    print('\n    Total params: %.2fM No' % (sum(p.numel() for p in model.parameters())/1000000.0))
    print('    Total trainable params: %.0f No' % (sum(p.numel() for p in model.parameters() if p.requires_grad)))

    # define loss function (criterion)
    criterion = nn.CrossEntropyLoss(reduction='mean')
    criterion.to(device)

    # optimizer definition:
    if _config['optimizer'] == 'SGD' or _config['optimizer'] == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=_config['lr'], momentum=_config['momentum'])
    
    elif _config['optimizer'] == 'Adam' or _config['optimizer'] == 'ADAM' or _config['optimizer'] == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=_config['lr'])
    
    else:
        raise ValueError("optimizer is not selected properly. For now, just 'SGD' and 'Adam' are available")

    # # Train and val
    start_epoch=1
    best_acc = 0.
    to_save=Path('/data/reza/checkpoint/place_informative')
    to_save.mkdir(parents=True,exist_ok=True)
    lr_scheduler = StepLR(optimizer, step_size=_config['lr_step'], gamma=_config['lr_gamma'])

    for epoch in range(start_epoch, _config['epochs']+start_epoch):
        print(f'\nEpoch: [{epoch} | {_config["epochs"]}] ')

        # train the model
        train_loss, train_acc = train_one_epoch(train_loader, model, criterion, optimizer, epoch, 
                                                                        device=device, phase='train')
        lr_scheduler.step()
        val_loss, val_acc= train_one_epoch(val_loader, model, criterion, optimizer, epoch, 
                                                                device=device, phase='val')

        # writing the scalars
            # tensorboard
        if _config['log_records'] == True:
            writer.add_scalars('loss', {'train': train_loss,
                                            'validation': val_loss}, epoch)
            writer.flush()
                # Accuracy
            writer.add_scalars('accuracy', {'train': train_acc,
                                            'validation': val_acc}, epoch)
            writer.flush()

            # sacred
            _run.log_scalar('train_loss', train_loss, epoch)
            _run.log_scalar('val_loss', val_loss, epoch)
            _run.log_scalar('train_acc', train_acc.item(), epoch)
            _run.log_scalar('val_acc', val_acc.item(), epoch)

        if val_acc > best_acc:
            print('Acuuracy improved. Saving the best model...')
            best_acc = val_acc
            torch.save(
                {
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                },
                    to_save / f'{_config["model_name"]}_epoch-{epoch}_acc-{val_acc:.3f}_{datetime.datetime.now().strftime("%Y%m%d")}.pth'
                    )
        else: 
            print('Accuracy does not improved')
        print(f'Best Acc: {best_acc:.4f}')

    if _config['log_records'] == True:
        writer.close()



@ex.automain
def main(_config, _run):
    train(_config, _run)

##TO DO
# 
#
