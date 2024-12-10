import copy
import torch
import types
import math
import numpy as np
from scipy import stats
import torch.nn.functional as F
import cvxpy as cvx


def average_weights_weighted(w, avg_weight):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    weight = torch.tensor(avg_weight)
    agg_w = weight / (weight.sum(dim=0))
    for key in w_avg.keys():
        # w_avg[key] = torch.zeros_like(w_avg[key])
        w_avg[key] = torch.zeros_like(w_avg[key]).float()  # change some int64 to float32
        for i in range(len(w)):
            # w_avg[key] += agg_w[i]*w[i][key]
            w_avg[key] += agg_w[i] * (w[i][key].float())
        # w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


def get_parameter_values(model):
    parameter = torch.cat([torch.reshape(param.data, (-1,)) for param in model.parameters()]).clone().detach()
    return parameter


def gaussian_noise(data_shape, clip, sigma, device=None):
    """
    Gaussian noise
    """
    return torch.normal(0, sigma * clip, data_shape).to(device)
