from datetime import datetime
from pathlib import Path
import hydra
from hydra.utils import to_absolute_path
import sys
from torchsummary import summary
from sacred import Experiment
from place_informative.src.models.factory import ModelBuild

date = datetime.now().strftime('%Y_%m_%d-%H_%M')

# specify config file
current_dir = Path(to_absolute_path(__file__))
config_name = str(Path(current_dir).resolve().parent.joinpath('config', 'config.yaml'))

#model summary
ex = Experiment('check_model')
ex.add_config(config_name)

@ex.capture
def model_summary(_config):
    """
    Script to check the model. 
    Model will be defined based on src/model/factory.py.
    It is enable to be shown in terminal or to be write in txt file by using the flag summary.summary_to_file=True.
    """
    # specify config items
    model_name = _config['model_name']
    num_classes = _config['num_classes']
    img_size = _config['img_size']
    batch_size = _config['batch_size']
    all_layers = _config['all_layers']
    summary_to_file = _config['summary_to_file']
    
    # specify summary output file
    output_dir = Path(current_dir).resolve().parents[2].joinpath('reports', 'model_summary')
    output_dir.mkdir(parents=True, exist_ok=True)
    modelsummary = output_dir.joinpath(f'model_summary_{model_name}_{date}.txt')

    # Model definition 
    model = ModelBuild(_config)

    # Flag to show the output in terminal or write to txt file
    summary_flag = summary_to_file

    # Summary
    if summary_flag == False:
        summary(model, input_size=(3, img_size, img_size), batch_size=batch_size, device='cpu')
    else:
        print(f'Model summary is generating and saving as ... \n{str(modelsummary)}')

        sys.stdout = open(str(modelsummary), 'w')
        summary(model, input_size=(3, img_size, img_size), batch_size=batch_size, device='cpu')
        sys.stdout.close()

@ex.automain
def my_main(_config):
    model_summary(_config)