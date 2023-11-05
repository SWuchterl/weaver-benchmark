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
        save_grad_inputs = False,
        pair_input_dim = len(data_config.input_dicts['pf_vectors']),
        pair_extra_dim = 0,
        embed_dims = [128, 256, 128],
        pair_embed_dims = [64, 64, 64],
        block_params = None,
        cls_block_params={'dropout': 0.05, 'attn_dropout': 0.05, 'activation_dropout': 0.05},
        num_heads = kwargs.get('num_heads',8),
        num_layers = kwargs.get('num_layers',8),
        num_cls_layers = kwargs.get('num_cls_layers',2),
        remove_self_pair = kwargs.get('remove_self_pair',True),
        use_pre_activation_pair = kwargs.get('use_pre_activation_pair',True),
        activation = kwargs.get('activation','gelu'),
        trim = kwargs.get('use_trim',True),
        for_inference = kwargs.get('for_inference',False),
        use_amp = kwargs.get('use_amp',False),
        split_da = kwargs.get('split_da',True),
        split_reg = kwargs.get('split_reg',True),
        fc_params = [(256, 0.1), (128, 0.1), (96, 0.1), (64, 0.1)],
        fc_da_params = [(128, 0.1), (96, 0.1), (64, 0.1)],
        alpha_grad = kwargs.get('alpha_grad',1)
    );

    model = ParticleTransformerTagger(**cfg)

    model_info = {
        'input_names':list(data_config.input_names),
        'input_shapes':{k:((1,) + s[1:]) for k, s in data_config.input_shapes.items()},
        'output_names':['output'],
        'dynamic_axes':{**{k:{0:'N', 2:'n_' + k.split('_')[0]} for k in data_config.input_names}, **{'output':{0:'N'}}},
        }

    return model, model_info

# https://github.com/yiftachbeer/mmd_loss_pytorch/tree/master
class RBF(nn.Module):
    def __init__(self, n_kernels=5, mul_factor=2.0, bandwidth=None):
        super().__init__()
        self.bandwidth_multipliers = mul_factor ** (torch.arange(n_kernels) - n_kernels // 2)
        self.bandwidth = bandwidth

    def get_bandwidth(self, L2_distances):
        if self.bandwidth is None:
            n_samples = L2_distances.shape[0]
            return L2_distances.data.sum() / (n_samples ** 2 - n_samples)

        return self.bandwidth

    def forward(self, X):
        L2_distances = torch.cdist(X, X) ** 2
        L2_distances = L2_distances.to(X.device,non_blocking=True);
        self.bandwidth_multipliers = self.bandwidth_multipliers.to(X.device,non_blocking=True);
        return torch.exp(-L2_distances[None, ...]  / (self.get_bandwidth(L2_distances) * self.bandwidth_multipliers)[:, None, None]).sum(dim=0)
    
## MMDLoss                                                                                                                                                                                             
class MMDLoss(nn.Module):

    def __init__(self, kernel=RBF()):
        super().__init__()
        self.kernel = kernel

    def forward(self, X, Y):
        K = self.kernel(torch.vstack([X, Y]).to(X.device,non_blocking=True))
        X_size = X.shape[0]
        XX = K[:X_size, :X_size].mean()
        XY = K[:X_size, X_size:].mean()
        YY = K[X_size:, X_size:].mean()
        return XX - 2 * XY + YY

class CrossEntropyLogCoshLossDomainAttack(torch.nn.L1Loss):
    __constants__ = ['reduction','loss_reg','loss_res','quantiles','loss_da','domain_weight','domain_dim','loss_attack','use_mmd_loss']

    def __init__(self, 
                 reduction: str = 'mean',
                 loss_reg: float = 1., 
                 loss_res: float = 1., 
                 loss_da: float = 1., 
                 loss_attack: float = 1.,
                 use_mmd_loss: bool = False,
                 quantiles: list = [],
                 domain_weight: list = [],
                 domain_dim: list = []
             ) -> None:
        super(CrossEntropyLogCoshLossDomainAttack, self).__init__(None, None, reduction)
        self.loss_reg = loss_reg;
        self.loss_res = loss_res;
        self.loss_da = loss_da;
        self.loss_attack = loss_attack;
        self.quantiles = quantiles;
        self.domain_weight = domain_weight;
        self.domain_dim = domain_dim;
        self.use_mmd_loss =  use_mmd_loss;
        if self.use_mmd_loss:
            self.MMDLoss = MMDLoss();
        else:
            self.MMDLoss = False;
        
    def forward(self, 
                input_cat: Tensor, y_cat: Tensor, 
                input_reg: Tensor, y_reg: Tensor, 
                input_domain: Tensor, y_domain: Tensor, y_domain_check: Tensor,
                input_cat_attack: Tensor = torch.Tensor(), input_cat_ref: Tensor = torch.Tensor()) -> Tensor:

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
                if torch.is_tensor(loss_quant): loss_quant = loss_quant.mean();
                if torch.is_tensor(loss_mean): loss_mean = loss_mean.mean();
                elif self.reduction == 'sum':
                    if torch.is_tensor(loss_quant): loss_quant = loss_quant.sum();
                if torch.is_tensor(loss_mean): loss_mean = loss_mean.sum();
            ## composition
            loss_reg = self.loss_reg*loss_mean+self.loss_res*loss_quant;

        ## domain terms
        loss_domain = 0;
        if input_domain.nelement():
            ## just one domain region
            if not self.domain_weight or len(self.domain_weight) == 1:
                loss_domain = self.loss_da*torch.nn.functional.cross_entropy(input_domain,y_domain,reduction=self.reduction);
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
                loss_domain *= self.loss_da;

        ## attack term
        loss_attack = 0;
        if input_cat_attack.nelement() and input_cat_ref.nelement():
            if self.use_mmd_loss:
                loss_attack = self.loss_attack*self.MMDLoss(torch.softmax(input_cat_ref,dim=1),torch.softmax(input_cat_attack,dim=1));
            else:
                input_cat_attack = torch.log_softmax(input_cat_attack,dim=1);
                input_cat_ref  = torch.softmax(input_cat_ref,dim=1);
                loss_attack = torch.nn.functional.kl_div(input=input_cat_attack,target=input_cat_ref,reduction='none');
                if self.reduction == 'mean':
                    loss_attack = self.loss_attack*loss_attack.mean();
                elif self.reduction == 'sum':
                    loss_attack = self.loss_attack*loss_attack.sum();
        return loss_cat+loss_reg+loss_domain+loss_attack, loss_cat, loss_reg, loss_domain, loss_attack;
    
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

    return CrossEntropyLogCoshLossDomainAttack(
        reduction=kwargs.get('reduction','mean'),
        loss_reg=kwargs.get('loss_reg',1),
        loss_res=kwargs.get('loss_res',1),
        loss_da=kwargs.get('loss_da',1),
        loss_attack=kwargs.get('loss_attack',1),
        use_mmd_loss=kwargs.get('use_mmd_loss',False),
        quantiles=quantiles,
        domain_weight=wdomain,
        domain_dim=ldomain
    );
