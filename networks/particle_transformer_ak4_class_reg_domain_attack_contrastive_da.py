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
        ## node feature block
        pf_input_dim = len(data_config.input_dicts['pf_features']),
        sv_input_dim = len(data_config.input_dicts['sv_features']),
        lt_input_dim = len(data_config.input_dicts['lt_features']),
        num_classes = num_classes,
        num_targets = num_targets,
        num_domains = num_domains,
        ## edge feature block
        pair_input_dim = len(data_config.input_dicts['pf_vectors']),
        pair_extra_dim = 0,
        remove_self_pair = kwargs.get('remove_self_pair',True),
        ## Embeddings
        embed_dims = [128, 256, 128],
        pair_embed_dims = [64, 64, 64],
        trim = kwargs.get('use_trim',True),
        activation = kwargs.get('activation','gelu'),
        ## Transformer block
        num_heads = kwargs.get('num_heads',8),
        num_layers = kwargs.get('num_layers',8),
        block_params = None,
        cls_block_params={'dropout': 0.05, 'attn_dropout': 0.05, 'activation_dropout': 0.05},
        num_cls_layers = kwargs.get('num_cls_layers',2),
        use_pre_activation_pair = kwargs.get('use_pre_activation_pair',True),
        ## Gradient step
        save_grad_inputs = False,
        use_amp = kwargs.get('use_amp',False),
        ## Post CLs parameters
        for_inference = kwargs.get('for_inference',False),
        fc_params = [(256, 0.1), (128, 0.1), (96, 0.1), (64, 0.1)],
        split_reg = kwargs.get('split_reg',True),
        ## DA parameters
        alpha_grad = kwargs.get('alpha_grad',1),
        split_da_outputs = kwargs.get('split_da',True),        
        fc_domain_params = [(128, 0.1), (96, 0.1), (64, 0.1)],
        ## Contrastive
        fc_contrastive_params = [(256, 0.05)],
        use_contrastive_domain = kwargs.get('use_contrastive_domain',False)
    );

    model = ParticleTransformerTagger(**cfg)

    model_info = {
        'input_names':list(data_config.input_names),
        'input_shapes':{k:((1,) + s[1:]) for k, s in data_config.input_shapes.items()},
        'output_names':['output'],
        'dynamic_axes':{**{k:{0:'N', 2:'n_' + k.split('_')[0]} for k in data_config.input_names}, **{'output':{0:'N'}}},
        }

    return model, model_info


class CrossEntropyContrastiveRegDomainAttack(torch.nn.L1Loss):
    __constants__ = ['reduction','select_label','loss_reg','loss_res','quantiles','loss_da','domain_weight','domain_dim','loss_attack','loss_cont','use_cont_domain','temperature']
    def __init__(self, 
                 reduction: str = 'mean',
                 loss_reg: float = 1., 
                 loss_res: float = 1., 
                 loss_da: float = 1., 
                 loss_attack: float = 1.,
                 loss_cont: float = 1.,
                 select_label: bool = False,
                 use_cont_domain: bool = False,
                 temperature: float = 0.1,
                 quantiles: list = [],
                 domain_weight: list = [],
                 domain_dim: list = []
             ) -> None:
        super(CrossEntropyContrastiveRegDomainAttack, self).__init__(None, None, reduction)
        self.loss_reg = loss_reg;
        self.loss_res = loss_res;
        self.loss_da  = loss_da;
        self.loss_attack = loss_attack;
        self.loss_cont = loss_cont;
        self.temperature = temperature;
        self.select_label = select_label;
        self.use_cont_domain = use_cont_domain;
        self.quantiles = quantiles;
        self.domain_weight = domain_weight;
        self.domain_dim = domain_dim;
        
    def forward(self, 
                input_cat: Tensor, y_cat: Tensor, 
                input_reg: Tensor, y_reg: Tensor, 
                input_domain: Tensor, y_domain: Tensor, y_domain_check: Tensor,
                input_cat_attack: Tensor = torch.Tensor(), input_cat_ref: Tensor = torch.Tensor(),
                input_cont: Tensor = torch.Tensor(), input_cont_domain: Tensor = torch.Tensor(),
                ) -> Tensor:


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
            if self.select_label: ## build KL divergence only on the score of y-cat type
                input_cat_attack = torch.softmax(input_cat_attack,dim=1);
                input_cat_ref  = torch.softmax(input_cat_ref,dim=1);
                loss_attack = torch.nn.functional.mse_loss(input=input_cat_attack.gather(1,y_cat.view(-1,1)),target=input_cat_ref.gather(1,y_cat.view(-1,1)),reduction='none');
            else:
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
            logits_cont = torch.nn.functional.normalize(input_cont, dim=1)
            logits_cont = torch.div(torch.matmul(logits_cont,logits_cont.permute(1,0)),self.temperature);
            logits_mask = torch.zeros(input_cont.shape).float().to(logits_cont.device,non_blocking=True);
            r, c = y_cat.view(-1,1).shape;
            logits_mask[torch.arange(r).reshape(-1,1).repeat(1,c).flatten(),y_cat.flatten()] = 1;
            logits_mask = torch.matmul(logits_mask,logits_mask.permute(1,0))
            logits_mask /= torch.sum(logits_mask,dim=1,keepdims=True)
            loss_contrastive = self.loss_cont*torch.nn.functional.cross_entropy(logits_cont,logits_mask,reduction=self.reduction);
            ## domain cont part
            if self.use_cont_domain and input_cont_domain.nelement():
                logits_cont_domain = torch.nn.functional.normalize(input_cont_domain, dim=1)
                logits_cont_domain = torch.div(torch.matmul(logits_cont_domain,logits_cont_domain.permute(1,0)),self.temperature);
                logits_mask_domain = torch.zeros(input_cont_domain.shape).float().to(logits_cont_domain.device,non_blocking=True);                
                r, c = y_domain.view(-1,1).shape;
                logits_mask_domain[torch.arange(r).reshape(-1,1).repeat(1,c).flatten(),y_domain.flatten()] = 1;
                logits_mask_domain = torch.matmul(logits_mask_domain,logits_mask_domain.permute(1,0))
                logits_mask_domain /= torch.sum(logits_mask_domain,dim=1,keepdims=True)
                loss_contrastive += self.loss_cont*torch.nn.functional.cross_entropy(logits_cont_domain,logits_mask_domain,reduction=self.reduction);
            
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
        quantiles=quantiles,
        domain_weight=wdomain,
        domain_dim=ldomain,
        select_label=kwargs.get('select_label',False),
        use_cont_domain=kwargs.get('use_contrastive_domain',False),
        temperature=kwargs.get('temperature',0.1),
    );
