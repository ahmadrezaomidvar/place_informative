from datetime import datetime
from pathlib import Path
import hydra
from hydra.utils import to_absolute_path
import sys
from torchsummary import summary
from place_informative.src.models.factory import ModelBuild

date = datetime.now().strftime('%Y_%m_%d-%H_%M')

# specify config file
current_dir = Path(to_absolute_path(__file__))
config_name = str(Path(current_dir).resolve().parent.joinpath('config', 'config.yaml'))

#model summary
@hydra.main(config_name=config_name)
def model_summary(cfg):
    """
    Script to check the model. 
    Model will be defined based on src/model/factory.py.
    It is enable to be shown in terminal or to be write in txt file by using the flag summary.summary_to_file=True.
    """
    # specify summary output file
    output_dir = Path(current_dir).resolve().parents[2].joinpath('reports', 'model_summary')
    output_dir.mkdir(parents=True, exist_ok=True)
    modelsummary = output_dir.joinpath(f'model_summary_{cfg.model.model_name}_{date}.txt')

    # Model definition 
    model = ModelBuild(cfg)

    # Flag to show the output in terminal or write to txt file
    summary_flag = cfg.summary.summary_to_file

    # Summary
    if summary_flag == False:
        summary(model, input_size=(3, cfg.model.img_size, cfg.model.img_size), batch_size=cfg.train.batch_size, device='cpu')
    else:
        print(f'Model summary is generating and saving as ... \n{str(modelsummary)}')

        sys.stdout = open(str(modelsummary), 'w')
        summary(model, input_size=(3, cfg.model.img_size, cfg.model.img_size), batch_size=cfg.train.batch_size, device='cpu')
        sys.stdout.close()

if __name__ == '__main__':
    model_summary()