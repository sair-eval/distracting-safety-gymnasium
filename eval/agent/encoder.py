# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file of
# https://github.com/facebookresearch/deep_bisim4control. Adapted for SAIR evaluation.

import torch
import torch.nn as nn


class PixelEncoder(nn.Module):
    """Convolutional encoder of stacked pixel observations (H, W, C) -> latent of size feature_dim."""

    def __init__(self, obs_shape, feature_dim, num_layers=4, num_filters=32, stride=None):
        super().__init__()
        assert len(obs_shape) == 3
        self.feature_dim = feature_dim
        self.num_layers = num_layers
        self.convs = nn.ModuleList([nn.Conv2d(obs_shape[2], num_filters, 3, stride=2)])
        for _ in range(num_layers - 1):
            self.convs.append(nn.Conv2d(num_filters, num_filters, 3, stride=1))
        h, w = obs_shape[0], obs_shape[1]
        for layer in self.convs:
            h = (h + 2 * layer.padding[0] - layer.kernel_size[0]) // layer.stride[0] + 1
            w = (w + 2 * layer.padding[1] - layer.kernel_size[1]) // layer.stride[1] + 1
        self.fc = nn.Linear(num_filters * h * w, feature_dim)
        self.ln = nn.LayerNorm(feature_dim)

    def forward_conv(self, obs):
        obs = obs / 255.
        if obs.ndim == 4:
            obs = obs.permute(0, 3, 1, 2)          # (B, H, W, C) -> (B, C, H, W)
        conv = torch.relu(self.convs[0](obs))
        for i in range(1, self.num_layers):
            conv = torch.relu(self.convs[i](conv))
        return conv.reshape(conv.size(0), -1)

    def forward(self, obs, detach=False):
        h = self.forward_conv(obs)
        if detach:
            h = h.detach()
        return self.ln(self.fc(h))


def make_encoder(encoder_type, obs_shape, feature_dim, num_layers, num_filters, stride):
    assert encoder_type == 'pixel', f'only pixel encoders are released, got {encoder_type!r}'
    return PixelEncoder(obs_shape, feature_dim, num_layers, num_filters, stride)
