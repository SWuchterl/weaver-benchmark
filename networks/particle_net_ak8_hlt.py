import numpy as np
import math
import torch
from torch import Tensor
from nn.model.ParticleNet import ParticleNetTagger

def get_model(data_config, **kwargs):

    ## input numer of point features to EdgeConvBlock                                                                                                                                                 
    point_features = 48;
    ## convoluational layers in EdgeConvBlock and kNN                                                                                                                                                 
    conv_params = [
        (16, (192, 144, 96)),
        (12, (192, 144, 96)),
        (8, (192, 144, 96)),
        ]
    ## use fusion layer for edge-conv block                                                                                                                                                           
    use_fusion = True
    ## fully connected output layers                                                                                                                                                                  
    fc_params = [
        (192, 0.1),
        (144, 0.1),
        (96, 0.1),
    ]

    ## classes and features
    pf_features_dims = len(data_config.input_dicts['pf_features'])
    sv_features_dims = len(data_config.input_dicts['sv_features'])
    num_classes = len(data_config.label_value);

    model = ParticleNetTagger(pf_features_dims=pf_features_dims, 
                                     sv_features_dims=sv_features_dims, 
                                     num_classes=num_classes,
                                     num_targets=0,
                                     conv_params=conv_params, 
                                     fc_params=fc_params,
                                     input_dims=point_features, 
                                     use_fusion=use_fusion,
                                     use_fts_bn=kwargs.get('use_fts_bn', False),
                                     use_counts=kwargs.get('use_counts', True),
                                     pf_input_dropout=kwargs.get('pf_input_dropout', None),
                                     sv_input_dropout=kwargs.get('sv_input_dropout', None),
                                     for_inference=kwargs.get('for_inference', False),
                                     alpha_grad=kwargs.get('alpha_grad',1)
                                 )

    model_info = {
        'input_names':list(data_config.input_names),
        'input_shapes':{k:((1,) + s[1:]) for k, s in data_config.input_shapes.items()},
        'output_names':['output'],
        'dynamic_axes':{**{k:{0:'N', 2:'n_' + k.split('_')[0]} for k in data_config.input_names}, **{'output':{0:'N'}}},
        }

    return model, model_info


def get_loss(data_config, **kwargs):

    return torch.nn.CrossEntropyLoss()
