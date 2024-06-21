import os
import torch
from weaver.utils.logger import _logger
from weaver.utils.import_tools import import_module
from torch import Tensor
import math

ParT = import_module(
    os.path.join(os.path.dirname(__file__), 'ParticleTransformerReg.py'), 'ParT')
# from ParticleTransformerReg import ParticleTransformerTagger
# from ParticleTransformerReg import ParticleTransformer as ParticleTransformerTagger_


def get_model(data_config, **kwargs):

    cfg = dict(
        num_classes=len(data_config.label_value),
        num_targets = len(data_config.target_value),
        # network configurations
        pair_input_dim=4,
        remove_self_pair=False,
        use_pre_norm_pair=False,
        embed_dims=[128, 128, 128],
        # embed_dims=[32, 32],
        pair_embed_dims=[64, 64, 64],
        num_heads=8,
        num_layers=8,
        num_cls_layers=2,
        block_params=None,
        # cls_block_params={'dropout': 0, 'attn_dropout': 0, 'activation_dropout': 0},
        cls_block_params={'dropout': 0.1, 'attn_dropout': 0.1, 'activation_dropout': 0.1},
        fc_params=[],
        activation='gelu',
        # misc
        trim=True,
        for_inference=False,
    )

    if 'lep_features' in data_config.input_dicts:
        # 1L and 2L
        cfg.update(
            pf_input_dim=len(data_config.input_dicts['jet_features']),
            sv_input_dim=len(data_config.input_dicts['lep_features']),
        )
        ParticleTransformer = ParT.ParticleTransformerTagger
    else:
        # 0L
        cfg.update(
            input_dim=len(data_config.input_dicts['jet_features']),
        )
        ParticleTransformer = ParT.ParticleTransformer
        # ParticleTransformer = ParticleTransformerTagger_

    cfg.update(**kwargs)
    _logger.info('Model config: %s' % str(cfg))

    model = ParticleTransformer(**cfg)

    model_info = {
        'input_names': list(data_config.input_names),
        'input_shapes': {k: ((1,) + s[1:]) for k, s in data_config.input_shapes.items()},
        'output_names': ['softmax'],
        'dynamic_axes': {**{k: {0: 'N', 2: 'n_' + k.split('_')[0]} for k in data_config.input_names}, **{'softmax': {0: 'N'}}},
    }

    return model, model_info


class CrossEntropyLogCoshLoss(torch.nn.L1Loss):
    __constants__ = ['reduction','nclass','ntarget','loss_lambda']

    def __init__(self, reduction: str = 'mean', nclass: int = 1, ntarget: int = 1, loss_lambda: float = 1.) -> None:
        super(CrossEntropyLogCoshLoss, self).__init__(None, None, reduction)
        self.nclass = nclass;
        self.ntarget = ntarget;
        self.loss_lambda = loss_lambda

    def forward(self, input: Tensor, y_cat: Tensor, inputReg: Tensor, y_reg: Tensor) -> Tensor:

        ## regression term
        # input_reg = input[:,self.nclass:self.nclass+self.ntarget].squeeze();
        input_reg = inputReg.squeeze();
        y_reg     = y_reg.squeeze();
        loss_reg  = (input_reg-y_reg)+torch.nn.functional.softplus(-2.*(input_reg-y_reg))-math.log(2);
        ## classification term
        input_cat = input[:,:self.nclass].squeeze();
        y_cat     = y_cat.squeeze().long();
        loss_cat  = torch.nn.functional.cross_entropy(input_cat,y_cat,reduction=self.reduction);
                
        # print ("input_reg")
        # print (input_reg)
        # print ("y_reg")
        # print (y_reg)

        ## final loss and pooling over batcc
        if self.reduction == 'none':            
            return loss_cat+self.loss_lambda*loss_reg, loss_cat, loss_reg*self.loss_lambda;
        elif self.reduction == 'mean':
            return loss_cat+self.loss_lambda*loss_reg.mean(), loss_cat, loss_reg.mean()*self.loss_lambda;
        elif self.reduction == 'sum':
            return loss_cat+self.loss_lambda*loss_reg.sum(), loss_cat, loss_reg.sum()*self.loss_lambda;


def get_loss(data_config, **kwargs):
    nclass  = len(data_config.label_value);
    ntarget = len(data_config.target_value);
    return CrossEntropyLogCoshLoss(reduction=kwargs.get('reduction','mean'),loss_lambda=kwargs.get('loss_lambda',1),nclass=nclass,ntarget=ntarget);