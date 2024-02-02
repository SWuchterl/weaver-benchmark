import numpy as np
import math
import torch
from torch import Tensor
from nn.model.ParticleTransformerCharged import ParticleTransformerTagger

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
        ## input tensor dimensions
        pf_ch_input_dim = len(data_config.input_dicts['pf_ch_features']),
        pf_neu_input_dim = len(data_config.input_dicts['pf_neu_features']),
        sv_input_dim = len(data_config.input_dicts['sv_features']),
        kaon_input_dim = len(data_config.input_dicts['k_features']),
        lambda_input_dim = len(data_config.input_dicts['l_features']),
        lt_input_dim = len(data_config.input_dicts['lt_features']),
        ## output dimensions
        num_classes = num_classes,
        num_targets = num_targets,
        num_domains = num_domains,
        ## embeddings
        embed_dims = [96, 192, 128],
        pair_input_dim = len(data_config.input_dicts['pf_ch_vectors']),
        pair_extra_dim = 0,        
        pair_embed_dims = [32, 96, 64],
        ## transformer parameters
        block_params = None,
        num_heads = kwargs.get('num_heads',8),
        num_layers = kwargs.get('num_layers',8),
        num_cls_layers = kwargs.get('num_cls_layers',2),
        cls_block_params={'dropout': 0.05, 'attn_dropout': 0.05, 'activation_dropout': 0.05},
        ## other options
        remove_self_pair = kwargs.get('remove_self_pair',True),
        use_pre_activation_pair = kwargs.get('use_pre_activation_pair',True),
        activation = kwargs.get('activation','gelu'),
        trim = kwargs.get('use_trim',True),
        use_amp = kwargs.get('use_amp',False),
        ## domain and attack
        alpha_grad = kwargs.get('alpha_grad',1),
        save_grad_inputs = False,
        split_da = kwargs.get('split_da',True),
        split_reg = kwargs.get('split_reg',True),
        ## final dense layers (nodes, dropout)
        fc_params = [(256, 0.1), (128, 0.1), (96, 0.1), (64, 0.1)],
        fc_da_params = [(128, 0.1), (96, 0.1), (64, 0.1)],
        fc_contrastive_params = [(128, 0.05)],
        for_inference = kwargs.get('for_inference',False),
        add_da_inference = kwargs.get('add_da_inference',False)
    );

    model = ParticleTransformerTagger(**cfg)

    model_info = {
        'input_names':list(data_config.input_names),
        'input_shapes':{k:((1,) + s[1:]) for k, s in data_config.input_shapes.items()},
        'output_names':['output'],
        'dynamic_axes':{**{k:{0:'N', 2:'n_' + k.split('_')[0]} for k in data_config.input_names}, **{'output':{0:'N'}}},
        }

    return model, model_info

####
class CrossEntropyContrastiveRegDomainAttack(torch.nn.L1Loss):
    __constants__ = ['reduction','loss_reg','loss_res','quantiles','loss_da','domain_weight','domain_dim','loss_attack','temperature','loss_cont']

    def __init__(self, 
                 reduction: str = 'mean',
                 loss_reg: float = 1., 
                 loss_res: float = 1., 
                 loss_da: float = 1., 
                 loss_attack: float = 1.,
                 loss_cont: float = 1.,
                 temperature: float = 0.1,
                 quantiles: list = [],
                 domain_weight: list = [],
                 domain_dim: list = []
             ) -> None:
        super(CrossEntropyContrastiveRegDomainAttack, self).__init__(None, None, reduction)
        self.loss_reg = loss_reg;
        self.temperature = temperature;
        self.loss_res = loss_res;
        self.loss_da = loss_da;
        self.loss_attack = loss_attack;
        self.loss_cont = loss_cont;
        self.quantiles = quantiles;
        self.domain_weight = domain_weight;
        self.domain_dim = domain_dim;
        
    def forward(self, 
                input_cat: Tensor, y_cat: Tensor, 
                input_reg: Tensor, y_reg: Tensor, 
                input_domain: Tensor, y_domain: Tensor, y_domain_check: Tensor,
                input_cat_attack: Tensor = torch.Tensor(), input_cat_ref: Tensor = torch.Tensor(),
                input_cont: Tensor = torch.Tensor(),
                ) -> Tensor:


        ## classification term
        loss_cat = 0;
        if input_cat.nelement():
            loss_cat = torch.nn.functional.cross_entropy(input_cat,y_cat,reduction=self.reduction);
            
        ## regression terms (nominal and quantiles)
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
            input_cat_attack = torch.log_softmax(input_cat_attack,dim=1);
            input_cat_ref  = torch.softmax(input_cat_ref,dim=1);
            loss_attack = torch.nn.functional.kl_div(input=input_cat_attack,target=input_cat_ref,reduction='none');
            if self.reduction == 'mean':
                loss_attack = self.loss_attack*loss_attack.mean();
            elif self.reduction == 'sum':
                loss_attack = self.loss_attack*loss_attack.sum();

        ## contrastive term
        loss_contrastive = 0;
        if input_cont.nelement():
            logits_contrastive = torch.nn.functional.normalize(input_cont, dim=1)
            logits_contrastive = torch.div(torch.matmul(logits_contrastive,logits_contrastive.permute(1,0)),self.temperature);
            logits_mask = torch.zeros(input_cat.shape).float().to(logits_contrastive.device,non_blocking=True);
            r, c = y_cat.view(-1,1).shape;
            logits_mask[torch.arange(r).reshape(-1,1).repeat(1,c).flatten(),y_cat.flatten()] = 1;
            logits_mask = torch.matmul(logits_mask,logits_mask.permute(1,0))
            logits_mask /= torch.sum(logits_mask,dim=1,keepdims=True)
            loss_contrastive = self.loss_cont*torch.nn.functional.cross_entropy(logits_contrastive,logits_mask,reduction=self.reduction);
            
        return loss_cat+loss_reg+loss_domain+loss_attack+loss_contrastive, loss_cat, loss_reg, loss_domain, loss_attack, loss_contrastive;

    
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

    return CrossEntropyContrastiveRegDomainAttack(
        reduction=kwargs.get('reduction','mean'),
        loss_reg=kwargs.get('loss_reg',1),
        loss_res=kwargs.get('loss_res',1),
        loss_da=kwargs.get('loss_da',1),
        loss_attack=kwargs.get('loss_attack',1),
        loss_cont=kwargs.get('loss_cont',1),
        temperature=kwargs.get('temperature',0.1),
        quantiles=quantiles,
        domain_weight=wdomain,
        domain_dim=ldomain
    );
