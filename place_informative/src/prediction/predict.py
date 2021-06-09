from pathlib import Path
from PIL import Image
import numpy as np
import hydra
from hydra.utils import to_absolute_path
import random
from tqdm import tqdm

import torch

from place_informative.src.models.factory import ModelBuild
from place_informative.data.generator import DatasetLoader
from place_informative.utils import transforms


config_name = str(Path(to_absolute_path(__file__)).resolve().parents[1].joinpath('models', 'config', 'config.yaml'))

# Device 
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print (f'\ntesting on {device} . . .')

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
    test_loader = DatasetLoader(cfg, 'test', transforms=T.get_transform('test')).Loader()

    # get pretrained model
    weight_file = cfg.train.weight_file

    # Load model and weights
    model = ModelBuild(cfg)
    checkpoint = torch.load(weight_file)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    print('\npretrained model loaded sucessfully...\n')

    # Prediction
    model.eval()
    prediction=[]
    running_corrects = 0        
    running_count = 0

    for inputs, targets in tqdm(test_loader, unit="batch"):
        inputs, targets = inputs.to(device), targets.to(device)
        
        with torch.no_grad():
            logits, probas = model(inputs)
            _, preds = torch.max(logits, 1)

        prediction.append(preds)
        batch_corrects = torch.sum(preds == targets.data)
        running_corrects += batch_corrects
        running_count += targets.size(0)

    test_acc = running_corrects.double() / running_count

    print(f'test acc: {test_acc:.4f}')
    print(prediction)




if __name__ == '__main__':
    main()

## TO DO
#
#