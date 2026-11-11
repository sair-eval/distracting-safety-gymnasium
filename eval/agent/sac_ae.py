# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file of
# https://github.com/facebookresearch/deep_bisim4control. Adapted for SAIR evaluation (inference only).

import torch
import torch.nn as nn
import torch.nn.functional as F

from agent.encoder import make_encoder


def squash(mu):
    return torch.tanh(mu)


class Actor(nn.Module):
    """Gaussian policy on the encoder's latent; evaluation uses the squashed mean action."""

    def __init__(self, obs_shape, action_shape, hidden_dim, encoder_type, encoder_feature_dim,
                 log_std_min, log_std_max, num_layers, num_filters, stride):
        super().__init__()
        self.encoder = make_encoder(encoder_type, obs_shape, encoder_feature_dim, num_layers, num_filters, stride)
        self.log_std_min, self.log_std_max = log_std_min, log_std_max
        self.trunk = nn.Sequential(
            nn.Linear(self.encoder.feature_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 2 * action_shape[0]))

    def forward(self, obs):
        mu, _ = self.trunk(self.encoder(obs)).chunk(2, dim=-1)
        return squash(mu)


class QFunction(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1))

    def forward(self, obs, action):
        return self.trunk(torch.cat([obs, action], dim=1))


class CostCritic(nn.Module):
    """Single-headed cost critic Q_C(o, a) with its own encoder copy; leaky ReLU output (slope 0.01)."""

    def __init__(self, obs_shape, action_shape, hidden_dim, encoder_type, encoder_feature_dim,
                 num_layers, num_filters, stride, leaky_slope=0.01):
        super().__init__()
        self.encoder = make_encoder(encoder_type, obs_shape, encoder_feature_dim, num_layers, num_filters, stride)
        self.Q = QFunction(self.encoder.feature_dim, action_shape[0], hidden_dim)
        self.leaky_slope = leaky_slope

    def forward(self, obs, action):
        return F.leaky_relu(self.Q(self.encoder(obs), action), negative_slope=self.leaky_slope)
