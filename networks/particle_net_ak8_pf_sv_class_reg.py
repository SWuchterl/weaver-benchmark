import numpy as np
import math
import torch
from torch import Tensor
from nn.model.ParticleNet import ParticleNetLostTrkTagger

def get_model(data_config, **kwargs):

    ## input numer of point features to EdgeConvBlock                                                                                                                                                 
    point_features = 48;
    ## convoluational layers in EdgeConvBlock and kNN                                                                                                                                                 
    conv_params = [
        (24, (224, 176, 128)),
        (16, (224, 176, 128)),
        (12, (224, 176, 128)),
        (8,  (224, 176, 128))
        ]
    ## use fusion layer for edge-conv block                                                                                                                                                           
    use_fusion = True
    ## fully connected output layers                                                                                                                                                                  
    fc_params = [
        (256, 0.1),
        (192, 0.1),
        (160, 0.1),
        (128, 0.1),
        (96, 0.1),
        (64, 0.1)
    ]

    ## classes and features
    pf_features_dims = len(data_config.input_dicts['pf_features'])
    sv_features_dims = len(data_config.input_dicts['sv_features'])
    lt_features_dims = len(data_config.input_dicts['lt_features'])
    num_classes = len(data_config.label_value);
    num_targets = len(data_config.target_value)

    model = ParticleNetLostTrkTagger(pf_features_dims=pf_features_dims, 
                                     sv_features_dims=sv_features_dims, 
                                     lt_features_dims=lt_features_dims,
                                     num_classes=num_classes,
                                     num_targets=num_targets,
                                     conv_params=conv_params, 
                                     fc_params=fc_params,
                                     input_dims=point_features, 
                                     use_fusion=use_fusion,
                                     use_fts_bn=kwargs.get('use_fts_bn', False),
                                     use_counts=kwargs.get('use_counts', True),
                                     pf_input_dropout=kwargs.get('pf_input_dropout', None),
                                     sv_input_dropout=kwargs.get('sv_input_dropout', None),
                                     lt_input_dropout=kwargs.get('lt_input_dropout', None),
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


class CrossEntropyLogCoshLoss(torch.nn.L1Loss):
    __constants__ = ['reduction','loss_lambda','loss_gamma','quantiles']

    def __init__(self, reduction: str = 'mean', loss_lambda: float = 1., loss_gamma: float = 1., quantiles: list = []) -> None:
        super(CrossEntropyLogCoshLoss, self).__init__(None, None, reduction)
        self.loss_lambda = loss_lambda;
        self.loss_gamma = loss_gamma;
        self.quantiles = quantiles;

    def forward(self, input_cat: Tensor, y_cat: Tensor, input_reg: Tensor, y_reg: Tensor) -> Tensor:

        ## classification term                                                                                                                                                                    
        loss_cat  = torch.nn.functional.cross_entropy(input_cat,y_cat,reduction=self.reduction);

        ## regression terms                                                                                                                                                                          
        x_reg      = input_reg-y_reg;
        loss_mean  = 0;
        loss_quant = 0;
        for idx,q in enumerate(self.quantiles):
            if idx>0 or len(self.quantiles)>1:
                x_reg_eval = x_reg[:,idx]
            else:
                x_reg_eval = x_reg
            if q <= 0:
                loss_mean += x_reg_eval+torch.nn.functional.softplus(-2.*x_reg_eval)-math.log(2);
            else:
                loss_quant += q*x_reg_eval[:,idx]*torch.ge(x_reg_eval[:,idx],0)
                loss_quant += (q-1)*(x_reg_eval[:,idx])*torch.less(x_reg_eval[:,idx],0);

        if self.reduction == 'mean':
            loss_mean  = loss_mean.mean();
            loss_quant = loss_quant.mean();
        elif self.reduction == 'sum':
            loss_mean  = loss_mean.sum();
            loss_quant = loss_quant.sum();

        loss_reg = self.loss_lambda*loss_mean+self.loss_gamma*loss_quant;

        return loss_cat+loss_reg, loss_cat, loss_reg;
        

def get_loss(data_config, **kwargs):

    quantiles = data_config.target_quantile;
    return CrossEntropyLogCoshLoss(reduction=kwargs.get('reduction','mean'),
                                   loss_lambda=kwargs.get('loss_lambda',1),
                                   loss_gamma=kwargs.get('loss_gamma',1),
                                   quantiles=quantiles);
