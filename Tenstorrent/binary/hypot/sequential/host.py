# SPDX-License-Identifier: Apache-2.0
import torch
import ttnn
import ttnn._ttnn
from pathlib import Path

# A-1 Import your three building blocks
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent.parent.parent
sys.path.append(str(ROOT))

from kernels.composable.square.host import host as square_host
from kernels.composable.sqrt.host import host as sqrt_host
from kernels.composable.binary_add.host import add as add_host

import random

# Set seeds for reproducibility
torch.manual_seed(42)
random.seed(42)

def get_inputs(case: int):
    """
    Returns a square-ish tile-multiple shape (B, D) used for BOTH inputs A and B.
    """
    B = D = 32
    if case == 0:
        B = D = 1
    elif case == 1:
        B = 1; D = 2
    elif case == 2:
        B = 2; D = 1
    elif case == 3:
        B = 2; D = 2
    elif case == 4:
        B = D = 64
    return (B, D)

# A-2 Add a function stub
def hypot_host(input_a: ttnn.Tensor, input_b: ttnn.Tensor) -> ttnn.Tensor:
    # A-3 Allocate a_sq (device tensor)
    a_sq = ttnn.allocate_tensor_on_device(
        ttnn.Shape(input_a.shape), ttnn.bfloat16, ttnn.TILE_LAYOUT, input_a.device()
    )
    
    # A-4 Compute a_sq using square_host
    a_sq = square_host(input_a)
    
    # A-5 Allocate & compute b_sq
    b_sq = ttnn.allocate_tensor_on_device(
        ttnn.Shape(input_b.shape), ttnn.bfloat16, ttnn.TILE_LAYOUT, input_b.device()
    )
    b_sq = square_host(input_b)
    
    # A-6 Allocate c_sq
    c_sq = ttnn.allocate_tensor_on_device(
        ttnn.Shape(input_a.shape), ttnn.bfloat16, ttnn.TILE_LAYOUT, input_a.device()
    )
    
    # A-7 Compute c_sq = a_sq + b_sq
    c_sq = add_host(a_sq, b_sq)
    
    # A-8 Allocate final out
    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape(input_a.shape), ttnn.bfloat16, ttnn.TILE_LAYOUT, input_a.device()
    )
    
    # A-9 Compute out = sqrt(c_sq)
    out = sqrt_host(c_sq)
    
    # A-10 Free temporaries
    a_sq.deallocate()
    b_sq.deallocate()
    c_sq.deallocate()
    
    # A-11 Return
    return out

def main():
    dev = ttnn.open_device(device_id=0)
    case = 5
    size = get_inputs(case=case)

    # two independent inputs with the SAME shape
    A = (torch.rand(size) - 0.5).to(torch.bfloat16)
    B = (torch.rand(size) - 0.5).to(torch.bfloat16)

    # move to TT
    A_tt = ttnn.from_torch(A, device=dev, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    B_tt = ttnn.from_torch(B, device=dev, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    # run fused hypot pipeline
    Y_tt = hypot_host(A_tt, B_tt)
    Y = ttnn.to_torch(Y_tt, device=dev)

    # PyTorch reference
    ref = torch.hypot(A.float(), B.float()).to(torch.bfloat16)

    print("max_err:", torch.max(torch.abs(Y - ref)))
    print("allclose:", torch.allclose(Y, ref, rtol=1e-2, atol=1e-2))

if __name__ == "__main__":
    main()
