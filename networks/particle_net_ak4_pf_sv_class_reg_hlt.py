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
        (16, (128, 96, 64)),
        (14, (128, 96, 64)),
        (12, (128, 96, 64))
        ]
    ## use fusion layer for edge-conv block
    use_fusion = True
    ## fully connected output layers
    fc_params = [
        (128, 0.1),
        (96, 0.1),
        (64, 0.1),
    ]

    ## classes and features
    pf_features_dims = len(data_config.input_dicts['pf_features'])
    sv_features_dims = len(data_config.input_dicts['sv_features'])
    num_classes = len(data_config.label_value);
    num_targets = len(data_config.target_value)

    model = ParticleNetTagger(pf_features_dims=pf_features_dims, 
                                     sv_features_dims=sv_features_dims, 
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
        input_cat = input_cat.squeeze();
        y_cat     = y_cat.squeeze().long();
        loss_cat  = torch.nn.functional.cross_entropy(input_cat,y_cat,reduction=self.reduction);

        ## regression terms
        input_reg  = input_reg.squeeze();
        y_reg      = y_reg.squeeze();
        x_reg      = input_reg-y_reg;

        loss_mean  = None;
        loss_quant = None;

        for idx,q in enumerate(self.quantiles):
            x_reg_eval = x_reg
            if idx>0 or len(self.quantiles)>1:
                x_reg_eval = x_reg[:,idx] # protect against dimension 1 tensors

            if q <= 0 and loss_mean is None:
                loss_mean = (x_reg_eval)+torch.nn.functional.softplus(-2.*(x_reg_eval))-math.log(2);
            elif q <= 0 and loss_mean is not None:
                loss_mean += (x_reg_eval)+torch.nn.functional.softplus(-2.*(x_reg_eval))-math.log(2);
            if q > 0 and loss_quant is None:
                loss_quant  = q*x_reg_eval*torch.ge(x_reg_eval,0)
                loss_quant += (q-1)*(x_reg_eval)*torch.less(x_reg_eval,0);
            elif q > 0 and loss_quant is not None:
                loss_quant += q*x_reg_eval*torch.ge(x_reg_eval,0)
                loss_quant += (q-1)*(x_reg_eval)*torch.less(x_reg_eval,0);                

        if loss_quant is not None and loss_mean is not None:
            loss_reg = self.loss_lambda*loss_mean+self.loss_gamma*loss_quant;
        elif loss_quant is None:
            loss_reg = self.loss_lambda*loss_mean;
        elif loss_reg is None:
            loss_reg = self.loss_gamma*loss_quant;

        ## over a batch
        if self.reduction == 'none':            
            return loss_cat+loss_reg, loss_cat, loss_reg;
        elif self.reduction == 'mean':
            return loss_cat+loss_reg.mean(), loss_cat, loss_reg.mean();
        elif self.reduction == 'sum':
            return loss_cat+loss_reg.sum(), loss_cat, loss_reg.sum();


def get_loss(data_config, **kwargs):

    quantiles = data_config.target_quantile;
    return CrossEntropyLogCoshLoss(reduction=kwargs.get('reduction','mean'),
                                   loss_lambda=kwargs.get('loss_lambda',1),
                                   loss_gamma=kwargs.get('loss_gamma',1),
                                   quantiles=quantiles);
