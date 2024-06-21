import os
import torch
from weaver.utils.logger import _logger
from weaver.utils.import_tools import import_module

ParticleNeXtTagger = import_module(
    os.path.join(os.path.dirname(__file__), 'ParticleNeXt.py'), 'PNXT').ParticleNeXtTagger


def get_model(data_config, **kwargs):
    cfg = dict(
        pf_features_dims=len(data_config.input_dicts['jet_features']),
        sv_features_dims=len(data_config.input_dicts['lep_features']),
        num_classes=len(data_config.label_value),
        layer_params=[(8, 256, None, 128), (8, 256, None, 128), (8, 256, None, 128)],  # noqa
        use_polarization_angle=True,
        trim=False,
    )
    cfg.update(**kwargs)
    _logger.info('Model config: %s' % str(cfg))

    model = ParticleNeXtTagger(**cfg)

    model_info = {
        'input_names': list(data_config.input_names),
        'input_shapes': {k: ((1,) + s[1:]) for k, s in data_config.input_shapes.items()},
        'output_names': ['softmax'],
        'dynamic_axes': {**{k: {0: 'N', 2: 'n_' + k.split('_')[0]} for k in data_config.input_names}, **{'softmax': {0: 'N'}}},
    }

    return model, model_info


def get_loss(data_config, **kwargs):
    return torch.nn.CrossEntropyLoss()
