import numpy as np
import math
import torch
from torch import Tensor
from nn.model.ParticleTransformerV2 import ParticleTransformerTagger
from nn.loss.mdmm import Constraint, EqConstraint, MaxConstraint, MinConstraint

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


############
class LossCategorization (torch.nn.L1Loss):
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction;
    def forward(self, inputs, classes):
        loss = 0;
        if inputs.nelement():
            loss = torch.nn.functional.cross_entropy(inputs,classes,reduction=self.reduction);
        return loss;
    
############
class LossRegression (torch.nn.L1Loss):
    def __init__(self, reduction: str = 'mean', quantiles: list = []):
        super().__init__()
        self.reduction = reduction;
        self.quantiles = quantiles;

    def forward(self, inputs):
        loss = 0;
        if inputs.nelement():
            for idx,q in enumerate(self.quantiles):
                if q >=0: continue;
                if idx>0 or len(self.quantiles)>1:
                    inputs_eval = inputs[:,idx]
                else:
                    inputs_eval = inputs
                loss += inputs_eval+torch.nn.functional.softplus(-2.*inputs_eval)-math.log(2);
            if self.reduction == 'mean':
                loss = loss.mean();
            elif self.reduction == 'sum':
                loss = loss.sum();
        return loss;
    
############
class LossQuantile (torch.nn.L1Loss):
    def __init__(self, reduction: str = 'mean', quantiles: list = []):
        super().__init__()
        self.reduction = reduction;
        self.quantiles = quantiles;

    def forward(self, inputs):
        loss = 0;
        if inputs.nelement():
            for idx,q in enumerate(self.quantiles):
                if q < 0: continue;
                if idx>0 or len(self.quantiles)>1:
                    inputs_eval = inputs[:,idx]
                else:
                    inputs_eval = inputs
                loss = q*inputs*torch.ge(inputs,0);
                loss += (q-1)*inputs*torch.less(inputs,0);
            if self.reduction == 'mean':
                loss = loss.mean();
            elif self.reduction == 'sum':
                loss = loss.sum();
        return loss;

############
class LossDomain (torch.nn.L1Loss):
    def __init__(self, reduction: str = 'mean', wdomain: list = [], ddomain: list = []):
        super().__init__()
        self.reduction = reduction;
        self.wdomain = wdomain;
        self.ddomain = ddomain;
        
    def forward(self, inputs, classes, cdomains):
        loss = 0;
        if inputs.nelement():
            if not self.wdomain or len(self.wdomain) == 1:
                loss = torch.nn.functional.cross_entropy(inputs,classes,reduction=self.reduction);
            else:
                for id,w in enumerate(self.wdomain):
                    id_dom  = id*self.ddomain[id];
                    d_check = cdomains[:,id]
                    indexes = d_check.nonzero();
                    d_val = inputs[indexes,id_dom:id_dom+self.ddomain[id]].squeeze();
                    d_pred = classes[indexes,id].squeeze();
                    if d_val.nelement():
                        loss += w*torch.nn.functional.cross_entropy(d_val,d_pred,reduction=self.reduction);
        return loss;

############
class LossAttack (torch.nn.L1Loss):
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction;

    def forward(self, inputs_r, inputs_a):
        loss = 0;
        if inputs_r.nelement() and inputs_a.nelement():
            inputs_a = torch.log_softmax(inputs_a,dim=1);
            inputs_r = torch.softmax(inputs_r,dim=1);
            loss = torch.nn.functional.kl_div(input=inputs_a,target=inputs_r,reduction='none');
            if self.reduction == 'mean':
                loss = loss.mean();
            elif self.reduction == 'sum':
                loss = loss.sum();
        return loss;


############
class CrossEntropyLogCoshLossDomainAttack(torch.nn.L1Loss):
    __constants__ = ['reduction','mdmm_max','mdmm_reg_scale','mdmm_q_scale','mdmm_da_scale','mdmm_attack_scale','mdmm_damp','quantiles','domain_weight','domain_dim',]
    def __init__(self, 
                 reduction: str = 'mean',
                 mdmm_max: float = 1.,
                 mdmm_reg_scale: float = 1.,
                 mdmm_q_scale: float = 1.,
                 mdmm_da_scale: float = 1.,
                 mdmm_attack_scale: float = 1.,
                 mdmm_damp: float = 1.,
                 quantiles: list = [],
                 domain_weight: list = [],
                 domain_dim: list = [],
             ) -> None:

        super(CrossEntropyLogCoshLossDomainAttack, self).__init__(None, None, reduction)
        ## inputs needed
        self.reduction = reduction
        self.quantiles = quantiles;
        self.domain_weight = domain_weight;
        self.domain_dim = domain_dim;
        self.mdmm_max = mdmm_max;
        self.mdmm_damp = mdmm_damp;
        self.mdmm_reg_scale = mdmm_reg_scale;
        self.mdmm_q_scale = mdmm_q_scale;
        self.mdmm_da_scale = mdmm_da_scale;
        self.mdmm_attack_scale = mdmm_attack_scale;
        ## losses 
        self.loss_class = LossCategorization(reduction=self.reduction);
        self.loss_reg = LossRegression(reduction=self.reduction,quantiles=self.quantiles);
        self.loss_quant = LossQuantile(reduction=self.reduction,quantiles=self.quantiles);
        self.loss_domain = LossDomain(reduction=self.reduction,wdomain=self.domain_weight,ddomain=self.domain_dim);
        self.loss_attack = LossAttack(reduction=self.reduction);
        ## constraint
        self.constraint_reg = EqConstraint(self.loss_reg,vale=self.mdmm_value,scale=self.mdmm_reg_scale,damping=self.mdmm_damp);
        self.constraint_quant = EqConstraint(self.loss_quant,value=self.mdmm_value,scale=self.mdmm_q_scale,damping=self.mdmm_damp);
        self.constraint_domain = EqConstraint(self.loss_domain,value=self.mdmm_value,scale=self.mdmm_da_scale,damping=self.mdmm_damp);
        self.constraint_attack = EqConstraint(self.loss_attack,value=self.mdmm_value,scale=self.mdmm_attack_scale,damping=self.mdmm_damp);
        self.constraints = [self.constraint_reg,self.constraint_quant,self.constraint_domain,self.constraint_attack]
        self.lambdas = [c.lmbda for c in self.constraints];
        self.slacks = [c.slack for c in self.constraints if hasattr(c, 'slack')];
                
    def forward(self, input_cat: Tensor, y_cat: Tensor, input_reg: Tensor, y_reg: Tensor, 
                input_domain: Tensor, y_domain: Tensor, y_domain_check: Tensor,
                input_cat_attack: Tensor = torch.Tensor(), input_cat_ref: Tensor = torch.Tensor()) -> Tensor:

        ## classification 
        total_loss = 0;
        loss_class = 0;
        if input_cat.nelement() and y_cat.nelement():
            loss_class = self.loss_class(input_cat,y_cat);
            total_loss += loss_class;
        ## regression
        loss_reg = 0;
        loss_quant = 0;
        if input_reg.nelement() and y_reg.nelement():
            loss_reg = self.constraint_reg([input_reg-y_reg]).value;
            loss_quant = self.constraint_quant([input_reg-y_reg]).value;
            total_loss += loss_reg+loss_quant;
            loss_reg   += loss_quant
        ## domain
        loss_domain = 0;
        if input_domain.nelement() and y_domain.nelement() and y_domain_check.nelement():
            loss_domain = self.constraint_domain([input_domain,y_domain,y_domain_check]).value;
            total_loss += loss_domain;
        ## attack
        loss_attack = 0;
        if input_cat_attack.nelement() and input_cat_ref.nelement():
            loss_attack = self.constraint_attack([input_cat_ref,input_cat_attack]).value;
            total_loss += loss_attack
        return total_loss,loss_class,loss_reg,loss_domain,loss_attack

    
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
        mdmm_value=kwargs.get('mdmm_value',1.),
        mdmm_damp=kwargs.get('mdmm_damp',1.),
        mdmm_reg_scale=kwargs.get('mdmm_reg_scale',1.),
        mdmm_q_scale=kwargs.get('mdmm_q_scale',1.),
        mdmm_da_scale=kwargs.get('mdmm_da_scale',1.),
        mdmm_attack_scale=kwargs.get('mdmm_attack_scale',1.),
        quantiles=quantiles,
        domain_weight=wdomain,
        domain_dim=ldomain
    );
          
