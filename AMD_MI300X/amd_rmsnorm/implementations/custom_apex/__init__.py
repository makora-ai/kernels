import numbers

import torch
from torch.nn import Parameter, init

from . import C
from .. import implementation


def _cast_if_autocast_enabled(*args):
    if not torch.is_autocast_enabled():
        return args
    else:
        return torch.amp.autocast_mode._cast(args, 'cuda', torch.get_autocast_gpu_dtype())


class FusedRMSNormAffineFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, normalized_shape, eps, residual=None, training=True):
        ctx.normalized_shape = normalized_shape
        ctx.eps = eps
        ctx.add_residual = True if residual is not None else False
        ctx.training = training
        input_ = input.contiguous()
        weight_ = weight.contiguous()
        residual_ = residual.contiguous() if ctx.add_residual else None

        if ctx.add_residual:
            out, *extra = C.rms_norm_residual(input_, residual_, normalized_shape, weight_, eps, training)
        else:
            # print('Custom fwd', input_.dtype, input_.shape, ctx.normalized_shape, weight_.dtype, weight.shape, ctx.eps)
            out, *extra = C.rms_norm(input_, normalized_shape, weight_, eps, training)

        if ctx.training:
            if ctx.add_residual:
                ctx.save_for_backward(weight_, *extra)
            else:
                ctx.save_for_backward(weight_, input_, *extra)

        return out

    @staticmethod
    def backward(ctx, grad_output):
        weight_, input, invvar = ctx.saved_tensors

        grad_input = grad_weight = grad_residual = None

        if ctx.add_residual:
            grad_input, grad_residual, grad_weight = C.rms_norm_residual_backward(
                    grad_output.contiguous(),
                    invvar,
                    input,
                    ctx.normalized_shape,
                    weight_,
                    ctx.eps
                )
        else:
            # print('Custom bwd', grad_output.dtype, grad_output.shape, invvar.dtype, invvar.shape, input.dtype, input.shape)
            grad_input, grad_weight = C.rms_norm_backward(
                    grad_output.contiguous(),
                    invvar,
                    input,
                    ctx.normalized_shape,
                    weight_,
                    ctx.eps
                )

        return grad_input, grad_weight, None, None,  grad_residual, None



def fused_rms_norm_affine(input, weight, normalized_shape, eps=1e-6, residual=None, training=False):
    args = _cast_if_autocast_enabled(input, weight, normalized_shape, eps, residual, training)
    with torch.amp.autocast('cuda', enabled=False):
        return FusedRMSNormAffineFunction.apply(*args)


class FusedRMSNorm(torch.nn.Module):
    r"""Applies RMS Normalization over a mini-batch of inputs
    Currently only runs on cuda() tensors.
    .. math::
        y = \frac{x}{\mathrm{RMS}[x]} * \gamma
    The root-mean-square is calculated separately over the last
    certain number dimensions which have to be of the shape specified by
    :attr:`normalized_shape`.
    :math:`\gamma` is a learnable affine transform parameter of
    :attr:`normalized_shape` if :attr:`elementwise_affine` is ``True``.
    `epsilon` is added to the mean-square, then the root of the sum is taken.
    .. note::
        Unlike Batch Normalization and Instance Normalization, which applies
        scalar scale and bias for each entire channel/plane with the
        :attr:`affine` option, RMS Normalization applies per-element scale
        with :attr:`elementwise_affine`.
    This layer uses statistics computed from input data in both training and
    evaluation modes.
    Args:
        normalized_shape (int or list or torch.Size): input shape from an expected input
            of size
            .. math::
                [* \times \text{normalized}\_\text{shape}[0] \times \text{normalized}\_\text{shape}[1]
                    \times \ldots \times \text{normalized}\_\text{shape}[-1]]
            If a single integer is used, it is treated as a singleton list, and this module will
            normalize over the last dimension which is expected to be of that specific size.
        eps: a value added to the denominator for numerical stability. Default: 1e-5
        elementwise_affine: a boolean value that when set to ``True``, this module
            has learnable per-element affine parameters initialized to ones (for weights)
            and zeros (for biases). Default: ``True``.
    Shape:
        - Input: :math:`(N, *)`
        - Output: :math:`(N, *)` (same shape as input)
    Examples::
        >>> input = torch.randn(20, 5, 10, 10)
        >>> # With Learnable Parameters
        >>> m = apex.normalization.FusedRMSNorm(input.size()[1:])
        >>> # Without Learnable Parameters
        >>> m = apex.normalization.FusedRMSNorm(input.size()[1:], elementwise_affine=False)
        >>> # Normalize over last two dimensions
        >>> m = apex.normalization.FusedRMSNorm([10, 10])
        >>> # Normalize over last dimension of size 10
        >>> m = apex.normalization.FusedRMSNorm(10)
        >>> # Activating the module
        >>> output = m(input)
        >>> residual = torch.randn(20, 5, 10, 10)
        >>> output = m(input, residual)
    .. _`Root Mean Square Layer Normalization`: https://arxiv.org/pdf/1910.07467.pdf
    """

    def __init__(self, normalized_shape, eps=1e-5, training=True):
        super().__init__()

        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = torch.Size(normalized_shape)
        self.eps = eps
        self._training = training

        self.weight = Parameter(torch.empty(*normalized_shape))
        self.reset_parameters()

    def reset_parameters(self):
        init.ones_(self.weight)

    def forward(self, input, residual=None):
        return fused_rms_norm_affine(input, self.weight, self.normalized_shape, self.eps, residual, self._training)

    def extra_repr(self):
        return "{normalized_shape}, eps={eps}, ".format(**self.__dict__)


@implementation('custom-apex')
def get_impl(hidden_size, eps, training):
    return FusedRMSNorm(hidden_size, eps=eps, training=training)
