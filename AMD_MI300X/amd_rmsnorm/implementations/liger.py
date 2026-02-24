
from liger_kernel.transformers.rms_norm import LigerRMSNorm

from . import implementation


@implementation('liger-inplace')
def get_impl(hidden_size, eps, training):
    return LigerRMSNorm(hidden_size, eps=eps, in_place=True)


@implementation('liger')
def get_impl(hidden_size, eps, training):
    return LigerRMSNorm(hidden_size, eps=eps, in_place=False)
