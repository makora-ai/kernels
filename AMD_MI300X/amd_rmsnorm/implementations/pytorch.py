import torch
import torch.nn as nn


from . import implementation


@implementation('pytorch')
def get_impl(hidden_size, eps, training):
    return nn.RMSNorm(hidden_size, eps=eps)


@implementation('pytorch-compiled')
def get_compiled(hidden_size, eps, training):
    return torch.compile(nn.RMSNorm(hidden_size, eps=eps), mode='max-autotune-no-cudagraphs')
