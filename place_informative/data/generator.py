from pathlib import Path
from PIL import Image
import os
import pandas as pd
from hydra.utils import to_absolute_path
import hydra
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

from place_informative.utils import data_split, transforms


# Dataset
class CityData(Dataset):
    """
    dataset preparation
    num_classes will be 7. Tehran, Dubai, Doha, Doshanbe, Istanbul, Masghat and Baku are the cities considered for model training.


    Parameters
    ----------
    config parameters

    mode: string
        'train', 'valid' or 'test'. Option to prepare related dataset.
    
    transforms: Torchvision transform
        Default to None. 

    Returns
    ----------
    img: PIL.Image
        image to train/validate/predict
    label: int
        city label between 0 to 6
    """
    def __init__(self, cfg, mode, transforms=None):
        super(CityData, self).__init__()
        
        # Config parameters
        self.mode = mode
        self.img_size = cfg.model.img_size
        self.transforms = transforms

        # Num of classes for CityData dataset.
        self.num_classes = 7

        # data set
        data = data_split(cfg, self.mode)
        try:
            self.data_set = data.split()
        except:
            raise ValueError("dataset mode shall be selected properly e.g: 'train', 'valid' or 'test ")

        # load class map
        self.class_map = data.load_class_map()

    def __len__(self):
        return len(self.data_set)

    def __getitem__(self, index):

        # labels will be cities index based on class_map
        label = self.data_set[index].split(os.path.sep)[-2]
        label = np.array([key for key in self.class_map.keys() if self.class_map[key]==label], dtype=np.int64)
        label = torch.from_numpy(label).squeeze()
        
        img = Image.open(self.data_set[index])

        # data transform which will be as per transform class in utils.py
        if self.transforms is not None:
            img = self.transforms(img)

        return (img, label)


# Dataloader
class DatasetLoader():
    """
    General dataloader for prepared datasets


    Parameters
    ----------
    config parameters

    mode: string
        'train', 'valid' or 'test'. Option to prepare related dataset.
    
    transforms: Torchvision transform

    Returns
    ----------
    Iterative dataloader object
        
    """

    def __init__(self, cfg, mode, transforms):

        # Confiig parameters
        self.batch_size = cfg.train.batch_size
        self.cfg = cfg
        self.num_workers = cfg.train.num_workers

        self.mode = mode
        self.transforms = transforms

    def Loader(self):

        data_set = CityData(self.cfg, self.mode, self.transforms)

        # data_loader object
        data_loader = DataLoader(dataset=data_set, batch_size=self.batch_size, shuffle=True, 
                                    drop_last=False, num_workers=self.num_workers, pin_memory=False)

        return data_loader   

## TO DO
# 
# 



# config_name = str(Path(to_absolute_path(__file__)).resolve().parents[1].joinpath('src', 'models', 'config', 'config.yaml'))
# @hydra.main(config_name=config_name)
# def main(cfg):
#     # data_set = CityData(cfg, 'test', transforms=transforms(cfg).get_transform('test'))
#     # img, label = data_set[0]
#     # print(label, img.size())
#     DataLoader = DatasetLoader(cfg, 'train', transforms=transforms(cfg).get_transform('train')).Loader()
#     print(len(DataLoader))
#     for x, y in DataLoader:
#         print(x.size(), y.size())
#         break




# if __name__ == '__main__':
#     main()