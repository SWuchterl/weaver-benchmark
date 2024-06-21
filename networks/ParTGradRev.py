import os
import torch
from weaver.utils.logger import _logger
from weaver.utils.import_tools import import_module
from torch import Tensor
from nn.model.ParticleTransformer import ParticleTransformerTagger
from nn.model.ParticleTransformer import ParticleTransformer as ParticleTransformer_


def get_model(data_config, **kwargs):

    # print ("data_config",data_config)
    # print ("kwargs",kwargs)

    ## number of classes
    num_classes = len(data_config.label_value)

    ## number of domain labels in the various regions (one binary or multiclass per region)
    num_domains = []
    if type(data_config.label_domain_value) == dict:
        for dct in data_config.label_domain_value.values():
            num_domains.append(len(dct))
    else:
        num_domains.append(len(data_config.label_domain_value))

    fc_params_ = []
    fc_params_dropout_ = kwargs.get("fc_dropout", 0.0)
    maxNumber = kwargs.get("fc_params", None)
    numbers = []
    number = maxNumber
    if maxNumber:
        while number >=64:
            numbers.append(int(number))
            number = number/2.

    if kwargs.get("fc_params", None):
        fc_params_ = [(int(n),fc_params_dropout_) for n in numbers]
        
    fc_domain_params_ = []
    fc_domain_params_dropout_ = kwargs.get("fc_domain_dropout", 0.0)
    maxNumber_domain = kwargs.get("fc_domain_params", None)
    numbers_domain = []
    number_domain = maxNumber_domain
    if maxNumber_domain:
        while number_domain >=64:
            numbers_domain.append(int(number_domain))
            number_domain = number_domain/2.

    if kwargs.get("fc_domain_params", None):
        fc_domain_params_ = [(int(n_domain),fc_domain_params_dropout_) for n_domain in numbers_domain]

    dropout_ = kwargs.get("dropout", 0.)

    # print(kwargs.get("for_inference", False))

    cfg = dict(
        num_classes=num_classes,
        num_domains=num_domains,
        # network configurations
        pair_input_dim=4,
        remove_self_pair=False,
        # use_pre_norm_pair=False,
        embed_dims=[128, 128, 128],
        pair_embed_dims=[64, 64, 64],
        num_heads=8,
        num_layers=8,
        num_cls_layers=2,
        block_params=None,
        cls_block_params={'dropout': dropout_, 'attn_dropout': dropout_, 'activation_dropout': dropout_},
        # cls_block_params={'dropout': 0, 'attn_dropout': 0, 'activation_dropout': 0},
        # cls_block_params={'dropout': 0.15, 'attn_dropout': 0.15, 'activation_dropout': 0.15},
        # cls_block_params={'dropout': 0.1, 'attn_dropout': 0.1, 'activation_dropout': 0.1},
        # fc_params=[],
        # fc_params = [(96, 0.1), (64, 0.1)],
        # fc_domain_params = [(96, 0.1), (64, 0.1)],
        # fc_params =        [(256, 0.1), (128, 0.1), (96, 0.1), (64, 0.1)],
        fc_params = fc_params_,
        # fc_domain_params = [(256, 0.1), (128, 0.1), (96, 0.1), (64, 0.1)],
        fc_domain_params = fc_domain_params_,
        # fc_domain_params = [(128, 0.15), (96, 0.15), (64, 0.15)],
        # fc_domain_params = [(128, 0.15), (64, 0.15)],
        # fc_domain_params = [(64, 0.15)],
        # fc_params = [],
        # fc_domain_params = [(64, 0.1)],
        # fc_params = [],
        # fc_domain_params = [],
        activation= kwargs.get("activation", 'gelu'),
        # misc
        trim=True,
        # for_inference=False,
        for_inference=kwargs.get("for_inference", False),
        # alpha_grad = 0.2,
        # alpha_grad = 0.15,
        # alpha_grad = 0.1,
        alpha_grad = kwargs.get('alpha_grad', 0.1),
        # alpha_grad = 0.5,
        # alpha_grad = 0.75,
        split_domain_outputs = False,
    )

    if 'lep_features' in data_config.input_dicts:
        # 1L and 2L
        cfg.update(
            pf_input_dim=len(data_config.input_dicts['jet_features']),
            sv_input_dim=len(data_config.input_dicts['lep_features']),
        )
        ParticleTransformer = ParticleTransformerTagger
    else:
        # 0L
        cfg.update(
            input_dim=len(data_config.input_dicts['jet_features']),
        )
        # ParticleTransformer = ParT.ParticleTransformer
        ParticleTransformer = ParticleTransformer_

    # print (kwargs)
    # cfg.update(**kwargs)
    _logger.info('Model config: %s' % str(cfg))

    model = ParticleTransformer(**cfg)

    model_info = {
        'input_names': list(data_config.input_names),
        'input_shapes': {k: ((1,) + s[1:]) for k, s in data_config.input_shapes.items()},
        'output_names': ['softmax'],
        'dynamic_axes': {**{k: {0: 'N', 2: 'n_' + k.split('_')[0]} for k in data_config.input_names}, **{'softmax': {0: 'N'}}},
    }

    # print ("========================",cfg.for_inference)

    return model, model_info


# def get_loss(data_config, **kwargs):
#     return torch.nn.CrossEntropyLoss()

class CrossEntropyLogCoshLossDomain(torch.nn.L1Loss):
    __constants__ = ['reduction','loss_lambda','loss_gamma','quantiles','loss_kappa','domain_weight','domain_dim']

    def __init__(self, 
                 reduction: str = 'mean', 
                 loss_kappa: float = 1., 
                 domain_weight: list = [],
                 domain_dim: list = [],
             ) -> None:
        super(CrossEntropyLogCoshLossDomain, self).__init__(None, None, reduction)
        self.loss_kappa = loss_kappa
        self.domain_weight = domain_weight
        self.domain_dim = domain_dim

    def forward(self, 
                input_cat: Tensor, y_cat: Tensor, 
                input_domain: Tensor, y_domain: Tensor, y_domain_check: Tensor) -> Tensor:

        # print ("input_cat")
        # print (input_cat)
        # print ("y_cat")
        # print (y_cat)
        # print ("input_domain")
        # print (input_domain)
        # print ("y_domain")
        # print (y_domain)

        ## classification term
        loss_cat  = 0
        if input_cat.nelement():
            loss_cat = torch.nn.functional.cross_entropy(input_cat,y_cat,reduction=self.reduction)

        ## domain terms
        loss_domain    = 0
        if input_domain.nelement():
            ## just one domain region
            if not self.domain_weight or len(self.domain_weight) == 1:
                loss_domain = self.loss_kappa*torch.nn.functional.cross_entropy(input_domain,y_domain,reduction=self.reduction)
            else:
                ## more domain regions with different relative weights
                for id,w in enumerate(self.domain_weight):
                    id_dom  = id*self.domain_dim[id]
                    y_check = y_domain_check[:,id]
                    indexes = y_check.nonzero();                    
                    y_val   = input_domain[indexes,id_dom:id_dom+self.domain_dim[id]].squeeze()
                    y_pred  = y_domain[indexes,id].squeeze()
                    if y_val.nelement():
                        loss_domain += w*torch.nn.functional.cross_entropy(y_val,y_pred,reduction=self.reduction)
                loss_domain *= self.loss_kappa
            
        return loss_cat+loss_domain, loss_cat, loss_domain

def get_loss(data_config, **kwargs):

    ## number of domain regions
    wdomain = data_config.label_domain_loss_weight
    ## number of lables for cross entropy in each domain
    if type(data_config.label_domain_value) == dict:
        ldomain = [len(dct) if type(dct) == list else 1 for dct in data_config.label_domain_value.values()]
    else:
        ldomain = [len(data_config.label_domain_value)]

    return CrossEntropyLogCoshLossDomain(
        reduction=kwargs.get('reduction','mean'),
        loss_kappa=kwargs.get('loss_kappa',1),
        domain_weight=wdomain,
        domain_dim=ldomain
    )