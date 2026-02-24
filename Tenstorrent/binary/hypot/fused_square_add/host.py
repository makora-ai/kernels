# SPDX-License-Identifier: Apache-2.0
import torch
import ttnn
import ttnn._ttnn
from pathlib import Path
import random

# Set seeds for reproducibility
torch.manual_seed(42)
random.seed(42)

def load_file(path: str) -> str:
    with open(path, "r") as f:
        return f.read()

ROOT = Path.cwd()
KERNELS_DIR = ROOT / "kernels" / "binary" / "hypot" / "fused_square_add"

# Load kernel sources
compute_src = load_file(KERNELS_DIR / "compute.cpp")

# Load existing kernels from binary_add
binary_add_dir = ROOT / "kernels" / "composable" / "binary_add"
read_src = load_file(binary_add_dir / "read.cpp")
write_src = load_file(binary_add_dir / "write.cpp")

# Import sqrt host for the complete hypot operation
import sys
sys.path.append(str(ROOT))
from kernels.composable.sqrt.host import host as sqrt_host

def fused_square_add(input_tensor1: ttnn.Tensor, input_tensor2: ttnn.Tensor) -> ttnn.Tensor:
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
        kernel_source=compute_src,  # Use the fused compute kernel
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

def fused_square_add_then_sqrt(input_tensor1: ttnn.Tensor, input_tensor2: ttnn.Tensor) -> ttnn.Tensor:
    """
    Implements hypot using fused square+add kernel + separate sqrt host.
    This is the complete hypot operation: square(A) + square(B) in one kernel, then sqrt separately.
    """
    # First, get square(A) + square(B) using the fused kernel
    sum_squares = fused_square_add(input_tensor1, input_tensor2)
    
    # Then apply sqrt using the sqrt host
    result = sqrt_host(sum_squares)
    
    # Clean up intermediate tensor
    sum_squares.deallocate()
    
    return result

def get_inputs():
    torch.manual_seed(0)
    a = torch.rand((64, 64), dtype=torch.bfloat16)
    b = torch.rand((64, 64), dtype=torch.bfloat16)
    return a, b

def run():
    device = ttnn.open_device(device_id=0)
    a, b = get_inputs()
    c = ttnn.from_torch(a, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    d = ttnn.from_torch(b, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    # Test the complete hypot implementation (fused square+add + sqrt)
    res = fused_square_add_then_sqrt(c, d)
    res_torch = ttnn.to_torch(res, device=device)
    
    # Expected result: hypot(a, b) - complete hypot operation
    gold = torch.hypot(a.float(), b.float()).to(torch.bfloat16)

    print("Testing fused_square_add_then_sqrt (complete hypot):")
    print("max_err :", torch.max(torch.abs(res_torch - gold)))
    print("avg_err :", torch.mean(torch.abs(res_torch - gold)))
    print("allclose:", torch.allclose(res_torch, gold, rtol=1e-2, atol=1e-2))

if __name__ == "__main__":
    run()
