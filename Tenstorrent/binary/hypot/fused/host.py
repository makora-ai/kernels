# SPDX-License-Identifier: Apache-2.0
import torch
import ttnn
import ttnn._ttnn
from pathlib import Path
import sys
import random

# Set seeds for reproducibility
torch.manual_seed(42)
random.seed(42)

def load_file(path: str) -> str:
    with open(path, "r") as f:
        return f.read()

ROOT = Path.cwd()
KERNELS_DIR = ROOT / "kernels" / "binary" / "hypot" / "fused"

# Load kernel sources
compute_square_add_src = load_file(ROOT / "kernels" / "binary" / "hypot" / "fused_square_add" / "compute.cpp")
compute_hypot_src = load_file(KERNELS_DIR / "compute.cpp")

# Load existing kernels from binary_add
binary_add_dir = ROOT / "kernels" / "composable" / "binary_add"
read_src = load_file(binary_add_dir / "read.cpp")
write_src = load_file(binary_add_dir / "write.cpp")

# Load sqrt host for partial fusion test
sqrt_dir = ROOT / "kernels" / "composable" / "sqrt"
sys.path.append(str(ROOT))
from kernels.composable.sqrt.host import host as sqrt_host
from kernels.composable.square.host import host as square_host
from kernels.composable.binary_add.host import add as add_host
from kernels.binary.hypot.sequential.host import hypot_host as sequential_hypot_host

def fused_hypot(input_tensor1: ttnn.Tensor, input_tensor2: ttnn.Tensor, include_sqrt: bool = True) -> ttnn.Tensor:
    """
    Fused hypot implementation with feature flag to choose between:
    - include_sqrt=True: square(A) + square(B) + sqrt = hypot(A, B)
    - include_sqrt=False: square(A) + square(B) only
    """
    # Output tensor mirrors inputs
    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(input_tensor1.shape),
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        input_tensor1.device(),
    )

    # --- Tile count ---
    B, D = input_tensor1.shape
    
    # tiles count
    Mt = B // 32
    Nt = D // 32
    num_tiles = max(1, Mt * Nt)
    per_core_tile_cnt = num_tiles  # Keep naming consistent with square/sqrt wording
    
    # --- CB config (tile = 32x32 bf16) ---
    tile_bytes = 32 * 32 * 2  # bf16 = 2 bytes
    tiles_per_cb = 2
    cb_total = tiles_per_cb * tile_bytes
    cb_page_size = tile_bytes

    in1_cb, in2_cb, out_cb = 0, 1, 16
    scratch_cb_2, scratch_cb_3 = 2, 3  # Scratch CBs for square results
    core = ttnn.CoreCoord(0, 0)
    core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])

    # CB formats
    in1_cb_format = ttnn.CBFormatDescriptor(
        buffer_index=in1_cb, data_format=ttnn.bfloat16, page_size=cb_page_size
    )
    in2_cb_format = ttnn.CBFormatDescriptor(
        buffer_index=in2_cb, data_format=ttnn.bfloat16, page_size=cb_page_size
    )
    out_cb_format = ttnn.CBFormatDescriptor(
        buffer_index=out_cb, data_format=ttnn.bfloat16, page_size=cb_page_size
    )
    scratch_cb_2_format = ttnn.CBFormatDescriptor(
        buffer_index=scratch_cb_2, data_format=ttnn.bfloat16, page_size=cb_page_size
    )
    scratch_cb_3_format = ttnn.CBFormatDescriptor(
        buffer_index=scratch_cb_3, data_format=ttnn.bfloat16, page_size=cb_page_size
    )

    # CB descriptors
    in1_cb_desc = ttnn.CBDescriptor(
        total_size=cb_total, core_ranges=core_grid, format_descriptors=[in1_cb_format]
    )
    in2_cb_desc = ttnn.CBDescriptor(
        total_size=cb_total, core_ranges=core_grid, format_descriptors=[in2_cb_format]
    )
    out_cb_desc = ttnn.CBDescriptor(
        total_size=cb_total, core_ranges=core_grid, format_descriptors=[out_cb_format]
    )
    scratch_cb_2_desc = ttnn.CBDescriptor(
        total_size=cb_total, core_ranges=core_grid, format_descriptors=[scratch_cb_2_format]
    )
    scratch_cb_3_desc = ttnn.CBDescriptor(
        total_size=cb_total, core_ranges=core_grid, format_descriptors=[scratch_cb_3_format]
    )

    # Compile-time args
    reader_ct_args = ttnn.TensorAccessorArgs(input_tensor1).get_compile_time_args()
    reader_ct_args.extend(ttnn.TensorAccessorArgs(input_tensor2).get_compile_time_args())
    writer_ct_args = ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args()

    # Runtime args
    reader_rt_args = [[input_tensor1.buffer_address(), input_tensor2.buffer_address(), per_core_tile_cnt]]
    writer_rt_args = [[output_tensor.buffer_address(), num_tiles]]
    compute_ct = [num_tiles]  # per_core_tile_cnt as CT arg (index 0)
    compute_rt_args = []  # No runtime args for compute

    # Choose compute kernel based on feature flag
    compute_src = compute_hypot_src if include_sqrt else compute_square_add_src

    # Kernel descriptors
    reader_k = ttnn.KernelDescriptor(
        kernel_source=read_src,
        source_type=ttnn._ttnn.program_descriptor.SourceType.SOURCE_CODE,
        core_ranges=core_grid,
        compile_time_args=reader_ct_args,
        runtime_args=[reader_rt_args],
        config=ttnn.ReaderConfigDescriptor(),
    )
    compute_k = ttnn.KernelDescriptor(
        kernel_source=compute_src,  # Use the appropriate compute kernel
        source_type=ttnn._ttnn.program_descriptor.SourceType.SOURCE_CODE,
        core_ranges=core_grid,
        compile_time_args=compute_ct,
        runtime_args=compute_rt_args,
        config=ttnn.ComputeConfigDescriptor(),
    )
    writer_k = ttnn.KernelDescriptor(
        kernel_source=write_src,
        source_type=ttnn._ttnn.program_descriptor.SourceType.SOURCE_CODE,
        core_ranges=core_grid,
        compile_time_args=writer_ct_args,
        runtime_args=[writer_rt_args],
        config=ttnn.WriterConfigDescriptor(),
    )

    # Program
    program_descriptor = ttnn.ProgramDescriptor(
        kernels=[reader_k, compute_k, writer_k],
        semaphores=[],
        cbs=[in1_cb_desc, in2_cb_desc, out_cb_desc, scratch_cb_2_desc, scratch_cb_3_desc],
    )

    # Execute
    return ttnn.generic_op([input_tensor1, input_tensor2, output_tensor], program_descriptor)

def partial_fused_hypot(input_tensor1: ttnn.Tensor, input_tensor2: ttnn.Tensor) -> ttnn.Tensor:
    """
    Partially fused hypot implementation:
    - Uses fused square+add kernel to get A² + B²
    - Then uses sqrt_host to get sqrt(A² + B²) = hypot(A, B)
    """
    # First, get A² + B² using the fused square+add kernel
    sum_squares = fused_hypot(input_tensor1, input_tensor2, include_sqrt=False)
    
    # Then apply sqrt using the existing sqrt host
    result = sqrt_host(sum_squares)
    
    # Clean up intermediate tensor
    sum_squares.deallocate()
    
    return result

def sequential_hypot(input_tensor1: ttnn.Tensor, input_tensor2: ttnn.Tensor) -> ttnn.Tensor:
    """
    Sequential hypot implementation using the existing sequential host:
    - Uses the hypot_host function from sequential/host.py
    - Implements: square(A) + square(B) + sqrt = hypot(A, B) using separate kernels
    """
    return sequential_hypot_host(input_tensor1, input_tensor2)

def get_inputs(case: int = 0):
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
    elif case == 5:
        B = 9
        D = 26
    elif case == 6:
        B = random.randint(1, 64)
        D = random.randint(1, 64)

    return (B, D)

def test_square_add_only(A: torch.Tensor, B: torch.Tensor):
    """Test square+add only (no sqrt)"""
    print("  Testing square+add only...")
    device = ttnn.open_device(device_id=0)

    # move to TT
    A_tt = ttnn.from_torch(A, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    B_tt = ttnn.from_torch(B, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    # run fused square+add pipeline (no sqrt)
    Y_tt = fused_hypot(A_tt, B_tt, include_sqrt=False)
    Y = ttnn.to_torch(Y_tt, device=device)

    # PyTorch reference: square(A) + square(B)
    ref = torch.square(A) + torch.square(B)

    max_err = torch.max(torch.abs(Y - ref))
    avg_err = torch.mean(torch.abs(Y - ref))
    allclose = torch.allclose(Y, ref, rtol=1e-2, atol=1e-2)
    
    print(f"    Max error: {max_err:.6f}")
    print(f"    Avg error: {avg_err:.6f}")
    print(f"    Allclose:  {allclose}")
    
    return max_err, avg_err, allclose

def test_complete_hypot(A: torch.Tensor, B: torch.Tensor):
    """Test complete hypot (square+add+sqrt)"""
    print("  Testing complete hypot...")
    device = ttnn.open_device(device_id=0)

    # move to TT
    A_tt = ttnn.from_torch(A, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    B_tt = ttnn.from_torch(B, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    # run complete fused hypot pipeline
    Y_tt = fused_hypot(A_tt, B_tt, include_sqrt=True)
    Y = ttnn.to_torch(Y_tt, device=device)

    # PyTorch reference: hypot(A, B)
    ref = torch.hypot(A.float(), B.float()).to(torch.bfloat16)

    max_err = torch.max(torch.abs(Y - ref))
    avg_err = torch.mean(torch.abs(Y - ref))
    allclose = torch.allclose(Y, ref, rtol=1e-2, atol=1e-2)
    
    print(f"    Max error: {max_err:.6f}")
    print(f"    Avg error: {avg_err:.6f}")
    print(f"    Allclose:  {allclose}")
    
    return max_err, avg_err, allclose

def test_case(case: int):
    """Test a specific case with both square+add and complete hypot"""
    print(f"\n{'='*50}")
    print(f"Testing Case {case}")
    print(f"{'='*50}")
    
    size = get_inputs(case=case)
    print(f"Tensor shape: {size}")
    
    # Generate test tensors for this case
    torch.manual_seed(42)  # Fixed seed for reproducibility
    A = (torch.rand(size) - 0.5).to(torch.bfloat16)
    B = (torch.rand(size) - 0.5).to(torch.bfloat16)
    
    print(f"Input A range: [{A.min():.4f}, {A.max():.4f}]")
    print(f"Input B range: [{B.min():.4f}, {B.max():.4f}]")
    
    # Test both modes with the same tensors
    sq_add_max, sq_add_avg, sq_add_ok = test_square_add_only(A, B)
    print()
    hyp_max, hyp_avg, hyp_ok = test_complete_hypot(A, B)
    
    # Summary for this case
    print(f"\n  Case {case} Summary:")
    print(f"    Square+Add: max_err={sq_add_max:.6f}, allclose={sq_add_ok}")
    print(f"    Complete Hypot: max_err={hyp_max:.6f}, allclose={hyp_ok}")
    
    return {
        'case': case,
        'shape': size,
        'square_add': {'max_err': sq_add_max, 'avg_err': sq_add_avg, 'allclose': sq_add_ok},
        'hypot': {'max_err': hyp_max, 'avg_err': hyp_avg, 'allclose': hyp_ok}
    }

def run():
    """Run tests similar to the existing pattern"""
    device = ttnn.open_device(device_id=0)
    case = 5  # Default case, can be changed
    size = get_inputs(case=case)

    torch.manual_seed(42)
    
    # two independent inputs with the SAME shape
    A = (torch.rand(size) - 0.5).to(torch.bfloat16)
    B = (torch.rand(size) - 0.5).to(torch.bfloat16)

    # move to TT
    A_tt = ttnn.from_torch(A, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    B_tt = ttnn.from_torch(B, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    print(f"Testing case {case} with shape {size}")
    print(f"Input A range: [{A.min():.4f}, {A.max():.4f}]")
    print(f"Input B range: [{B.min():.4f}, {B.max():.4f}]")
    print()

    # Test square+add only
    # print("Testing square+add only...")
    # Y_sq_add_tt = fused_hypot(A_tt, B_tt, include_sqrt=False)
    # Y_sq_add = ttnn.to_torch(Y_sq_add_tt, device=device)
    # ref_sq_add = torch.square(A) + torch.square(B)
    
    # max_err_sq_add = torch.max(torch.abs(Y_sq_add - ref_sq_add))
    # avg_err_sq_add = torch.mean(torch.abs(Y_sq_add - ref_sq_add))
    # allclose_sq_add = torch.allclose(Y_sq_add, ref_sq_add, rtol=1e-2, atol=1e-2)
    
    # print("max_err:", max_err_sq_add)
    # print("avg_err:", avg_err_sq_add)
    # print("allclose:", allclose_sq_add)
    # print()

    # # Test partial fused hypot (square+add kernel + sqrt host)
    # print("Testing partial fused hypot (square+add + sqrt_host)...")
    # Y_partial_tt = partial_fused_hypot(A_tt, B_tt)
    # Y_partial = ttnn.to_torch(Y_partial_tt, device=device)
    # ref_partial = torch.hypot(A.float(), B.float()).to(torch.bfloat16)
    
    # max_err_partial = torch.max(torch.abs(Y_partial - ref_partial))
    # avg_err_partial = torch.mean(torch.abs(Y_partial - ref_partial))
    # allclose_partial = torch.allclose(Y_partial, ref_partial, rtol=1e-2, atol=1e-2)
    
    # print("max_err:", max_err_partial)
    # print("avg_err:", avg_err_partial)
    # print("allclose:", allclose_partial)
    # print()

    # Test complete hypot (fully fused)
    print("Testing complete hypot (fully fused)...")
    Y_hypot_tt = fused_hypot(A_tt, B_tt, include_sqrt=True)
    Y_hypot = ttnn.to_torch(Y_hypot_tt, device=device)
    ref_hypot = torch.hypot(A.float(), B.float()).to(torch.bfloat16)
    
    max_err_hypot = torch.max(torch.abs(Y_hypot - ref_hypot))
    avg_err_hypot = torch.mean(torch.abs(Y_hypot - ref_hypot))
    allclose_hypot = torch.allclose(Y_hypot, ref_hypot, rtol=1e-2, atol=1e-2)
    
    print("max_err:", max_err_hypot)
    print("avg_err:", avg_err_hypot)
    print("allclose:", allclose_hypot)
    print()

    # Test sequential hypot (three separate kernels)
    # print("Testing sequential hypot (square + add + sqrt)...")
    # Y_sequential_tt = sequential_hypot(A_tt, B_tt)
    # Y_sequential = ttnn.to_torch(Y_sequential_tt, device=device)
    # ref_sequential = torch.hypot(A.float(), B.float()).to(torch.bfloat16)
    
    # max_err_sequential = torch.max(torch.abs(Y_sequential - ref_sequential))
    # avg_err_sequential = torch.mean(torch.abs(Y_sequential - ref_sequential))
    # allclose_sequential = torch.allclose(Y_sequential, ref_sequential, rtol=1e-2, atol=1e-2)
    
    # print("max_err:", max_err_sequential)
    # print("avg_err:", avg_err_sequential)
    # print("allclose:", allclose_sequential)

def main():
    run()

if __name__ == "__main__":
    main()
