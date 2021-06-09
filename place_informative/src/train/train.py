import hydra
from pathlib import Path
from hydra.utils import to_absolute_path
import numpy as np
import random
from tqdm import tqdm
import datetime

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

@hydra.main(config_name=config_name)
def main(cfg):
    # Random seed
    if cfg.train.seed:
        seed=cfg.train.seed
        random.seed(seed) 
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)                                         
    ## Data loading code
    T = transforms(cfg)
    train_loader = DatasetLoader(cfg, 'train', transforms=T.get_transform('train')).Loader() 
    val_loader = DatasetLoader(cfg, 'valid', transforms=T.get_transform('valid')).Loader()
    test_loader = DatasetLoader(cfg, 'test', transforms=T.get_transform('test')).Loader()

    
    # create model
    model = ModelBuild(cfg)

    # SummaryWriter
    if cfg.tensorboard == True:
        log_dir = Path(to_absolute_path(__file__)).resolve().parents[2].joinpath('reports', 'tb_log', cfg.model.model_name, 
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
    if cfg.train.optimizer == 'SGD' or cfg.train.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=cfg.train.lr, momentum=cfg.train.momentum)
    
    elif cfg.train.optimizer == 'Adam' or cfg.train.optimizer == 'ADAM' or cfg.train.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=cfg.train.lr)
    
    else:
        raise ValueError("optimizer is not selected properly. For now, just 'SGD' and 'Adam' are available")

    # # Train and val
    start_epoch=1
    best_acc = 0.
    to_save=Path('/data/reza/checkpoint/place_informative')
    to_save.mkdir(parents=True,exist_ok=True)
    lr_scheduler = StepLR(optimizer, step_size=cfg.train.lr_step, gamma=cfg.train.lr_gamma)

    for epoch in range(start_epoch, cfg.train.epochs+start_epoch):
        print(f'\nEpoch: [{epoch} | {cfg.train.epochs}] ')

        # train the model
        train_loss, train_acc = train_one_epoch(train_loader, model, criterion, optimizer, epoch, 
                                                                        device=device, phase='train')
        lr_scheduler.step()
        val_loss, val_acc= train_one_epoch(val_loader, model, criterion, optimizer, epoch, 
                                                                device=device, phase='val')

        # writing the scalars
            # LOSS
        if cfg.tensorboard == True:
            writer.add_scalars('loss', {'train': train_loss,
                                            'validation': val_loss}, epoch)
            writer.flush()
                # Accuracy
            writer.add_scalars('accuracy', {'train': train_acc,
                                            'validation': val_acc}, epoch)
            writer.flush()

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
                    to_save / f'{cfg.model.model_name}_epoch-{epoch}_acc-{val_acc:.3f}_{datetime.datetime.now().strftime("%Y%m%d")}.pth'
                    )
        else: 
            print('Accuracy does not improved')
        print(f'Best Acc: {best_acc:.4f}')

    if cfg.tensorboard == True:
        writer.close()



if __name__ == '__main__':
    main()


##TO DO
# 
#
