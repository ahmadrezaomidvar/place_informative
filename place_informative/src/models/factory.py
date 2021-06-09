import torchvision.models as models
import torch.nn as nn
from hydra.utils import to_absolute_path
import hydra
from pathlib import Path
import torch


class ModelBuild(nn.Module):
    """
    Model factory
    Uses transfer learning with using the model specified in config file and loaded the pretrained weights.
    pretrained models can be found https://github.com/pytorch/vision/tree/master/torchvision/models.

    It is based on various parameters such as number of classes.
    For city data, num_classes will be 7.

    Weights can be freezed or be trainble based on option specified in config file.
    Last layers are changed based on the model and number of classes. This part shall be modified based on each model specified in config.


    Parameters
    ----------
    config parameters 


    Returns
    ----------
    logits: torch.tensor
        Output of fully connected layer. 
        logits will be in shape of [batch_size, num_classes]. 

    probas: torch.tensor
        Probability of logits.
        probas will be in shape of [batch_size, num_classes]. 

    """
    def __init__(self, cfg):
        super(ModelBuild, self).__init__()
        self.cfg = cfg
        self.model_name = cfg.model.model_name

        # Defining the number of classes.
        self.num_classes = 7

        # Models available in pytorch
        self.model_names = sorted(name for name in models.__dict__
                                     if name.islower() and not name.startswith('__')
                                      and callable(models.__dict__[name]))
        if f'{self.model_name}' in self.model_names:
            pass
        else:
            
           raise ValueError("""
           Model is not available in torchvision models, please select an available model.
           Check https://github.com/pytorch/vision/tree/master/torchvision/models for more details
            """)

        # loading pretrained model
        print("\n=> using pre-trained model '{}'".format(self.model_name))
        self.model = models.__dict__[f'{self.model_name}'](pretrained=True)

        # freezing the layers
        if self.cfg.train.all_layers == False:
            for param in self.model.parameters():
                param.requires_grad = False

        # changing the last layers
        ########################################### 
        '''
        The following part of code shall be changed for each model. 
        Use https://github.com/pytorch/vision/tree/master/torchvision/models to find out the last layer name 
        and spec to change the code based on it.
        '''
        ###########################################

        ########### ==> Start of the part to be modified based on model
        self.in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features=self.in_features, out_features=self.num_classes)
        # probas calculation layer
        self.softmax = nn.Softmax(dim=1)
        ########### <== End of the part to be modified based on model

    def forward(self, x):
        
        logits = self.model(x)
        probas = self.softmax(logits)

        return logits, probas



## TO DO
#
#


# from place_informative.data.generator import DatasetLoader          
# from place_informative.utils import transforms           
# config_name = str(Path(to_absolute_path(__file__)).resolve().parents[1].joinpath('models', 'config', 'config.yaml'))
# # Device 
# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# print (f'\ntraining on {device} . . .')

# @hydra.main(config_name=config_name)
# def main(cfg):
#     T = transforms(cfg)
#     model = ModelBuild(cfg)
#     print(model)
#     model.train()
#     train_loader = DatasetLoader(cfg, 'train', transforms=T.get_transform('train')).Loader() 
#     for inputs, targets in train_loader:
#         logits, probas = model(inputs)
#         print(logits, logits.size(), probas, probas.size())
#         break 
    





# if __name__ == '__main__':
#     main()