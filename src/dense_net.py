import torch
import torch.nn as nn
from torch import Tensor
import torch.utils.data as data
import numpy as np
from typing import Any, List, Tuple

class _DenseLayer(nn.Module):
    def __init__(self, num_neurons, layerNum, drop_rate=0.0, batch_norm=True):
        super().__init__()
        self.batch_norm = batch_norm

        self.add_module('lin{}'.format(layerNum), nn.Linear(num_neurons, num_neurons))
        self.add_module('tanh{}'.format(layerNum), nn.Tanh())
        if batch_norm:
            self.add_module('batch{}'.format(layerNum), nn.BatchNorm1d(num_neurons))
        self.add_module('drop{}'.format(layerNum), nn.Dropout(drop_rate))

    def forward(self, input: Tensor) -> Tensor:
        # TODO this might not work or might not be the most efficient option
        for module in self.children():
            input = module(input)
        return input

        ##if self.batch_norm:
        ##    new_features = self.drop0(self.batch0(self.tanh0(self.lin0(input))))
        ##else:
        ##    new_features = self.drop0(self.tanh0(self.lin0(input)))

        ##return new_features

        # bottleneck the incoming features into something that can go into the layer




class _DenseBlock(nn.ModuleDict):
    def __init__(self, num_layers, neurons_per_layer, drop_rate=0.0, batch_norm=True):
        super().__init__()
        for i in range(num_layers):
            layer = _DenseLayer(neurons_per_layer, i, drop_rate=drop_rate, batch_norm=batch_norm)

            self.add_module('denselayer{}'.format(i), layer)

    def forward(self, init_features: Tensor) -> Tensor:
        # This forward function iterates through layers and sums the inputs all
        # preceding layers output features into that layer
        features = init_features
        for name, layer in self.items():
            new_features = layer(features)
            features = features + new_features
        return features

class DenseNet(nn.Module):
    def __init__(self, num_input, num_output,
                 layers_per_block, neurons_per_blocklayer, drop_rate=0.0, batch_norm=True):
        super().__init__()

        # First linear layer
        self.add_module('linInp', nn.Linear(num_input, neurons_per_blocklayer[0]))
        self.add_module('tanhInp', nn.Tanh())
        if batch_norm:
            self.add_module('batchInp', nn.BatchNorm1d(neurons_per_blocklayer[0]))
        self.add_module('dropInp', nn.Dropout(drop_rate))

        # Dense blocks
        for i, (num_layers, num_neurons) in enumerate(zip(layers_per_block, neurons_per_blocklayer)):
            block = _DenseBlock(num_layers, num_neurons,
                                drop_rate=drop_rate, batch_norm=batch_norm)
            self.add_module('denseblock{}'.format(i), block)
            if i != len(layers_per_block) - 1:
                self.add_module('linTrans{}'.format(i), nn.Linear(num_neurons, neurons_per_blocklayer[i+1]))
                self.add_module('tanhTrans{}'.format(i), nn.Tanh())
                if batch_norm:
                    self.add_module('batchTrans{}'.format(i), nn.BatchNorm1d(neurons_per_blocklayer[i+1]))
                self.add_module('dropTrans{}'.format(i), nn.Dropout(drop_rate))

        # Final layer
        self.add_module('linF', nn.Linear(neurons_per_blocklayer[-1], num_output))
        self.add_module('softplus', nn.Softplus())

    def forward(self, input):
        # TODO this might not work or might not be the most efficient option
        for module in self.children():
            input = module(input)
        return input


class SimpleEncoder(nn.Module):
    def __init__(self, layer_sizes, drop_rate=0.2, batch_norm=False):
        super().__init__()
        for layerNum in range(len(layer_sizes)-1):
            self.add_module('lin{}'.format(layerNum), nn.Linear(layer_sizes[layerNum],
                                                                layer_sizes[layerNum+1]))
            if layerNum != len(layer_sizes)-2:
                # add a tanh if it is not the last layer
                self.add_module('tanh{}'.format(layerNum), nn.Tanh())
            if batch_norm:
                self.add_module('batch{}'.format(layerNum), nn.BatchNorm1d(layer_sizes[layerNum+1]))
            self.add_module('drop{}'.format(layerNum), nn.Dropout(drop_rate))

        self.add_module('softplus', nn.Softplus())

    def forward(self, input):
        for module in self.children():
            input = module(input)
        return input
