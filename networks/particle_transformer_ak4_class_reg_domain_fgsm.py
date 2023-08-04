import numpy as np
import math
import torch
from torch import Tensor
from nn.model.ParticleTransformerV2 import ParticleTransformerTagger

def get_model(data_config, **kwargs):

    ## number of classes
    num_classes = len(data_config.label_value)
    
    ## number of targets
    num_targets = 0;
    if type(data_config.target_value) == dict:
        num_targets = sum(len(dct) if type(dct) == list else 1 for dct in data_config.target_value.values())
    else:
        num_targets = len(data_config.target_value);

    ## number of domain labels in the various regions (one binary or multiclass per region)
    num_domains = [];
    if type(data_config.label_domain_value) == dict:
        for dct in data_config.label_domain_value.values():
            num_domains.append(len(dct))
    else:
        num_domains.append(len(data_config.label_domain_value));

    ## options                                                                                                                                                                                   
    cfg = dict(
        pf_input_dim = len(data_config.input_dicts['pf_features']),
        sv_input_dim = len(data_config.input_dicts['sv_features']),
        lt_input_dim = len(data_config.input_dicts['lt_features']),
        num_classes = num_classes,
        num_targets = num_targets,
        num_domains = num_domains,
        save_grad_inputs = True,
        pair_input_dim = len(data_config.input_dicts['pf_vectors']),
        pair_extra_dim = 0,
        embed_dims = [128, 256, 128],
        pair_embed_dims = [64, 64, 64],
        block_params = None,
        cls_block_params={'dropout': 0.05, 'attn_dropout': 0.05, 'activation_dropout': 0.05},
        num_heads = kwargs.get('num_heads',8),
        num_layers = kwargs.get('num_layers',8),
        num_cls_layers = kwargs.get('num_cls_layers',2),
        remove_self_pair = kwargs.get('remove_self_pair',False),
        use_pre_activation_pair = kwargs.get('use_pre_activation_pair',True),
        activation = kwargs.get('activation','gelu'),
        trim = kwargs.get('use_trim',True),
        for_inference = kwargs.get('for_inference',False),
        alpha_grad = kwargs.get('alpha_grad',1),
        use_amp = kwargs.get('use_amp',False),
        split_domain_outputs = kwargs.get('split_domain_outputs',False),
        fc_params = [(256, 0.1), (128, 0.1), (96, 0.1), (64, 0.1)],
        fc_domain_params = [(128, 0.1), (96, 0.1), (64, 0.1)]
    );

    model = ParticleTransformerTagger(**cfg)

    model_info = {
        'input_names':list(data_config.input_names),
        'input_shapes':{k:((1,) + s[1:]) for k, s in data_config.input_shapes.items()},
        'output_names':['output'],
        'dynamic_axes':{**{k:{0:'N', 2:'n_' + k.split('_')[0]} for k in data_config.input_names}, **{'output':{0:'N'}}},
        }

    return model, model_info


class CrossEntropyLogCoshLossDomainFgsm(torch.nn.L1Loss):
    __constants__ = ['reduction','loss_lambda','loss_gamma','quantiles','loss_kappa','domain_weight','domain_dim','loss_omega','loss_fgsm_type']

    def __init__(self, 
                 reduction: str = 'mean', 
                 loss_lambda: float = 1., 
                 loss_gamma: float = 1., 
                 loss_kappa: float = 1., 
                 loss_omega: float = 1.,
                 loss_fgsm_type: int = 0,
                 quantiles: list = [],
                 domain_weight: list = [],
                 domain_dim: list = []
             ) -> None:
        super(CrossEntropyLogCoshLossDomainFgsm, self).__init__(None, None, reduction)
        self.loss_lambda = loss_lambda;
        self.loss_gamma = loss_gamma;
        self.loss_kappa = loss_kappa;
        self.loss_omega = loss_omega;
        self.quantiles = quantiles;
        self.domain_weight = domain_weight;
        self.domain_dim = domain_dim;
        self.loss_fgsm_type = loss_fgsm_type;
        
    def forward(self, 
                input_cat: Tensor, y_cat: Tensor, 
                input_reg: Tensor, y_reg: Tensor, 
                input_domain: Tensor, y_domain: Tensor, y_domain_check: Tensor,
                input_cat_fgsm: Tensor, input_cat_ref: Tensor) -> Tensor:

        ## classification term
        loss_cat = 0;
        if input_cat.nelement():
            loss_cat = torch.nn.functional.cross_entropy(input_cat,y_cat,reduction=self.reduction);

        ## regression terms
        x_reg = input_reg-y_reg;        
        loss_mean = 0;
        loss_quant = 0;
        loss_reg = 0;
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
        loss_domain = 0;
        if input_domain.nelement():
            ## just one domain region
            if not self.domain_weight or len(self.domain_weight) == 1:
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

        ## fgsm term
        loss_fgsm = 0;
        if input_cat_fgsm.nelement() and input_cat_ref.nelement():
            if self.loss_fgsm_type == 1:
                loss_fgsm = self.loss_omega*torch.nn.functional.mse_loss(                    
                    input=torch.softmax(input_cat_fgsm,dim=1),
                    target=torch.softmax(input_cat_ref,dim=1),
                    reduction=self.reduction);
            elif self.loss_fgsm_type == 0 or self.loss_fgsm_type == -1:
                loss_fgsm = self.loss_omega*torch.nn.functional.kl_div(
                    input=torch.softmax(input_cat_fgsm,dim=1),
                    target=torch.softmax(input_cat_ref,dim=1),
                    log_target=True,reduction='batchmean' if self.reduction == "mean" else self.reduction).abs();
        return loss_cat+loss_reg+loss_domain+loss_fgsm, loss_cat, loss_reg, loss_domain, loss_fgsm;
    
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

    return CrossEntropyLogCoshLossDomainFgsm(
        reduction=kwargs.get('reduction','mean'),
        loss_lambda=kwargs.get('loss_lambda',1),
        loss_gamma=kwargs.get('loss_gamma',1),
        loss_kappa=kwargs.get('loss_kappa',1),
        loss_omega=kwargs.get('loss_omega',1),
        loss_fgsm_type=kwargs.get('loss_fgsm_type',0),
        quantiles=quantiles,
        domain_weight=wdomain,
        domain_dim=ldomain
    );
