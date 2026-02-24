

from apex.normalization import FusedRMSNorm

from . import implementation


@implementation('apex')
def get_impl(hidden_size, eps, training):
    return FusedRMSNorm(hidden_size, eps=eps)
