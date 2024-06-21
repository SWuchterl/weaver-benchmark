import torch
import torch.nn as nn
import numpy as np
from weaver.utils.logger import _logger


class MLP(nn.Module):

    def __init__(self,
                 input_dim,
                 num_classes,
                 # network configurations
                 fc_params=[(256, 0.1), (256, 0.1), (256, 0.1)],
                 add_bn=True,
                 for_inference=False,
                 **kwargs) -> None:
        super().__init__(**kwargs)

        fcs = [nn.BatchNorm1d(input_dim)]
        in_dim = input_dim
        for out_dim, drop_rate in fc_params:
            fcs.append(nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.ReLU(),
                nn.BatchNorm1d(out_dim) if add_bn else nn.Identity(),
                nn.Dropout(drop_rate)
            ))
            in_dim = out_dim
        fcs.append(nn.Linear(in_dim, num_classes))
        self.fc = nn.Sequential(*fcs)

        self.for_inference = for_inference

    def forward(self, jet_features, lep_features, event_features):
        # *_features: (batch_size, num_features, num_elements)
        x = torch.cat([jet_features.flatten(1), lep_features.flatten(1), event_features.flatten(1)], dim=1)
        # print('x:\n', x)
        output = self.fc(x)
        if self.for_inference:
            output = torch.softmax(output, dim=1)
        # print('output:\n', output)
        return output


def get_model(data_config, **kwargs):

    cfg = dict(
        input_dim=sum([np.prod(s[1:]) for s in data_config.input_shapes.values()]),
        num_classes=len(data_config.label_value),
        # network configurations
        fc_params=[(256, 0.1), (256, 0.1), (256, 0.1)],
        add_bn=True,
        for_inference=False,
    )

    cfg.update(**kwargs)
    _logger.info('Model config: %s' % str(cfg))

    model = MLP(**cfg)

    model_info = {
        'input_names': list(data_config.input_names),
        'input_shapes': {k: ((1,) + s[1:]) for k, s in data_config.input_shapes.items()},
        'output_names': ['softmax'],
        'dynamic_axes': {**{k: {0: 'N'} for k in data_config.input_names}, **{'softmax': {0: 'N'}}},
    }

    return model, model_info


def get_loss(data_config, **kwargs):
    return torch.nn.CrossEntropyLoss()
