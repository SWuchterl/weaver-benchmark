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
        (16, (256, 208, 176)),
        (14, (256, 208, 176)),
        (12, (256, 208, 176))
        ]
    ## use fusion layer for edge-conv block
    use_fusion = True
    ## fully connected output layers
    fc_params = [
        (224, 0.1),
        (192, 0.1),
        (160, 0.1),
        (128, 0.1),
        (96,  0.1),
        (64,  0.1)
    ]
    ## fully connected output layers
    fc_domain_params = [
        (224, 0.1),
        (192, 0.1),
        (160, 0.1),
        (128, 0.1),
        (96,  0.1),
        (64,  0.1)
    ]

    ## classes and features
    pf_features_dims = len(data_config.input_dicts['pf_features'])
    sv_features_dims = len(data_config.input_dicts['sv_features'])
    lt_features_dims = len(data_config.input_dicts['lt_features'])

    ## number of classes for the multi-class loss
    num_classes = len(data_config.label_value);
    ## number of targets
    if type(data_config.target_value) == dict:
        num_targets = sum(len(dct) if type(dct) == list else 1 for dct in data_config.target_value.values())
    else:
        num_targets = len(data_config.target_value);
    ## number of domain labels in the various regions (one binary or multiclass per region)
    if type(data_config.label_domain_value) == dict:
        num_domains = sum(len(dct) if type(dct) == list else 1 for dct in data_config.label_domain_value.values())
    else:
        num_domains = len(data_config.label_domain_value);

    model = ParticleNetLostTrkTagger(pf_features_dims=pf_features_dims, 
                                     sv_features_dims=sv_features_dims, 
                                     lt_features_dims=lt_features_dims, 
                                     num_classes=num_classes,
                                     num_targets=num_targets,
                                     num_domains=num_domains,
                                     conv_params=conv_params, 
                                     fc_params=fc_params,
                                     fc_domain_params=fc_domain_params,
                                     input_dims=point_features, 
                                     use_fusion=use_fusion,
                                     use_fts_bn=kwargs.get('use_fts_bn', False),
                                     use_counts=kwargs.get('use_counts', True),
                                     use_revgrad=kwargs.get('use_revgrad', True),
                                     pf_input_dropout=kwargs.get('pf_input_dropout', None),
                                     sv_input_dropout=kwargs.get('sv_input_dropout', None),
                                     lt_input_dropout=kwargs.get('lt_input_dropout', None),
                                     for_inference=kwargs.get('for_inference', False),
                                     alpha_grad=kwargs.get('alpha_grad', 1)
                                 )

    model_info = {
        'input_names':list(data_config.input_names),
        'input_shapes':{k:((1,) + s[1:]) for k, s in data_config.input_shapes.items()},
        'output_names':['output'],
        'dynamic_axes':{**{k:{0:'N', 2:'n_' + k.split('_')[0]} for k in data_config.input_names}, **{'output':{0:'N'}}},
        }

    return model, model_info


class CrossEntropyLogCoshLossDomain(torch.nn.L1Loss):
    __constants__ = ['reduction','loss_lambda','loss_gamma','quantiles','loss_kappa','domain_weight','domain_dim']

    def __init__(self, 
                 reduction: str = 'mean', 
                 loss_lambda: float = 1., 
                 loss_gamma: float = 1., 
                 loss_kappa: float = 1., 
                 quantiles: list = [],
                 domain_weight: list = [],
                 domain_dim: list = [],
             ) -> None:
        super(CrossEntropyLogCoshLossDomain, self).__init__(None, None, reduction)
        self.loss_lambda = loss_lambda;
        self.loss_gamma = loss_gamma;
        self.loss_kappa = loss_kappa;
        self.quantiles = quantiles;
        self.domain_weight = domain_weight;
        self.domain_dim = domain_dim;

    def forward(self, 
                input_cat: Tensor, y_cat: Tensor, 
                input_reg: Tensor, y_reg: Tensor, 
                input_domain: Tensor, y_domain: Tensor, y_domain_check: Tensor) -> Tensor:

        ## classification term
        loss_cat  = 0;
        if input_cat.nelement():
            loss_cat = torch.nn.functional.cross_entropy(input_cat,y_cat,reduction=self.reduction);

        ## regression terms
        x_reg      = input_reg-y_reg;        
        loss_mean  = 0;
        loss_quant = 0;
        loss_reg   = 0;
        if input_reg.nelement():
            ## compute loss
            for idx,q in enumerate(self.quantiles):
                if idx>0 or len(self.quantiles)>1:
                    x_reg_eval = x_reg[:,idx]
                else:
                    x_reg_eval = x_reg
                    if q <= 0:
                        loss_mean += x_reg_eval+torch.nn.functional.softplus(-2.*x_reg_eval)-math.log(2);
                    elif q > 0:
                        loss_quant += q*x_reg_eval*torch.ge(x_reg_eval,0);
                        loss_quant += (q-1)*x_reg_eval*torch.less(x_reg_eval,0);

            ## reduction
            if self.reduction == 'mean':
                loss_quant = loss_quant.mean();
                loss_mean = loss_mean.mean();
            elif self.reduction == 'sum':
                loss_quant = loss_quant.sum();
                loss_mean = loss_mean.sum();
            ## composition
            loss_reg = self.loss_lambda*loss_mean+self.loss_gamma*loss_quant;

        ## domain terms
        loss_domain    = 0;
        if input_domain.nelement():
            ## just one domain region
            if not self.domain_weight:
                loss_domain = self.loss_kappa*torch.nn.functional.cross_entropy(input_domain,y_domain,reduction=self.reduction);
            else:
                ## more domain regions with different relative weights
                for id,w in enumerate(self.domain_weight):
                    id_dom  = id*self.domain_dim[id];
                    y_check = y_domain_check[:,id]
                    indexes = y_check.nonzero();                    
                    y_val   = input_domain[indexes,id_dom:id_dom+self.domain_dim[id]].squeeze();
                    y_pred  = y_domain[indexes,id].squeeze();
                    if y_val.nelement():
                        loss_domain += w*torch.nn.functional.cross_entropy(y_val,y_pred,reduction=self.reduction);
                loss_domain *= self.loss_kappa;
            
        return loss_cat+loss_reg+loss_domain, loss_cat, loss_reg, loss_domain;

def get_loss(data_config, **kwargs):

    ## number of targets
    quantiles = data_config.target_quantile;
    ## number of domain regions
    wdomain = data_config.label_domain_loss_weight;
    ## number of lables for cross entropy in each domain
    if type(data_config.label_domain_value) == dict:
        ldomain = [len(dct) if type(dct) == list else 1 for dct in data_config.label_domain_value.values()]
    else:
        ldomain = [len(data_config.label_domain_value)];

    return CrossEntropyLogCoshLossDomain(
        reduction=kwargs.get('reduction','mean'),
        loss_lambda=kwargs.get('loss_lambda',1),
        loss_gamma=kwargs.get('loss_gamma',1),
        loss_kappa=kwargs.get('loss_kappa',1),
        quantiles=quantiles,
        domain_weight=wdomain,
        domain_dim=ldomain
    );
