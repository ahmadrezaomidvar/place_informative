from pathlib import Path
from PIL import Image
import numpy as np
import hydra
from hydra.utils import to_absolute_path
import random
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import json
import seaborn as sn
from sacred import Experiment
from sacred.observers import MongoObserver

import torch

from place_informative.src.models.factory import ModelBuild
from place_informative.data.generator import DatasetLoader
from place_informative.utils import transforms


config_name = str(Path(to_absolute_path(__file__)).resolve().parents[1].joinpath('models', 'config', 'config.yaml'))
ex = Experiment('urban_prediction')
ex.add_config(config_name)

# Device 
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print (f'\ntesting on {device} . . .')

@ex.capture
def pred(_config, _run):
    # Random seed
    if _config['seed']:
        seed=_config['seed']
        random.seed(seed) 
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)                                         
    ## Data loading code
    T = transforms(_config)
    test_loader = DatasetLoader(_config, 'test', transforms=T.get_transform('test')).Loader()

    # get pretrained model
    weight_file = _config['weight_file']

    # Load model and weights
    model = ModelBuild(_config)
    checkpoint = torch.load(weight_file)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    print('\npretrained model loaded sucessfully...\n')

    with open(f'{_config["root"]}/class_map.json') as json_file:
        class_map = json.load(json_file)    

    # Prediction
    model.eval()
    running_corrects = 0        
    running_count = 0

    targets_list = torch.zeros(0,dtype=torch.long, device='cpu')
    pred_list = torch.zeros(0,dtype=torch.long, device='cpu')
    for inputs, targets in tqdm(test_loader, unit="batch"):
        inputs, targets = inputs.to(device), targets.to(device)
        
        with torch.no_grad():
            logits, probas = model(inputs)
            _, preds = torch.max(logits, 1)

        targets_list=torch.cat([targets_list,targets.view(-1).cpu()])
        pred_list=torch.cat([pred_list,preds.view(-1).cpu()])

        batch_corrects = torch.sum(preds == targets.data)
        running_corrects += batch_corrects
        running_count += targets.size(0)

    conf_matx=confusion_matrix(targets_list.numpy(), pred_list.numpy())
    test_acc = running_corrects.double() / running_count
    print(f'test Acc: {test_acc:.4f}')

    print('Generating confusion matrix heat map...')

    df_cm = pd.DataFrame(conf_matx/np.sum(conf_matx), index = [i for i in class_map.values()], columns = [i for i in class_map.values()])
    plt.figure(figsize = (12,7))
    sn.heatmap(df_cm, annot=True)
    plt.ylabel('True city')
    plt.xlabel('Predicted city')   
    plt.savefig('output.png')



@ex.automain
def main(_config, _run):
    pred(_config, _run)

## TO DO
#
#