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
KERNELS_DIR = ROOT / "kernels" / "composable" / "sqrt"

READ_SRC_PATH    = KERNELS_DIR / "read.cpp"
WRITE_SRC_PATH   = KERNELS_DIR / "write.cpp"
COMPUTE_SRC_PATH = KERNELS_DIR / "compute.cpp"

read_tiles_src = load_file(READ_SRC_PATH)
write_tiles_src = load_file(WRITE_SRC_PATH)
compute_src = load_file(COMPUTE_SRC_PATH)


def host(input_tensor: ttnn.Tensor) -> ttnn.Tensor:
    # Output mirrors input (shape/dtype/layout/device)
    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(input_tensor.shape),
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        input_tensor.device(),
    )

    # --- CB config (tile = 32x32 bf16) ---
    tile_bytes = 32 * 32 * 2
    tiles_per_cb = 2
    cb_total = tiles_per_cb * tile_bytes
    cb_in, cb_out = 0, 16

    in_fmt  = ttnn.CBFormatDescriptor(buffer_index=cb_in,  data_format=ttnn.bfloat16, page_size=tile_bytes)
    out_fmt = ttnn.CBFormatDescriptor(buffer_index=cb_out, data_format=ttnn.bfloat16, page_size=tile_bytes)

    core = ttnn.CoreCoord(0, 0)
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])

    in_cb_desc  = ttnn.CBDescriptor(total_size=cb_total, core_ranges=grid, format_descriptors=[in_fmt])
    out_cb_desc = ttnn.CBDescriptor(total_size=cb_total, core_ranges=grid, format_descriptors=[out_fmt])

    # --- Accessor CT args ---
    reader_ct = ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args()
    writer_ct = ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args()

    # --- Tile count ---
    B, D = input_tensor.shape

    # tiles count
    Mt = B // 32
    Nt = D // 32
    num_tiles = max(1, Mt * Nt)

    # --- RT/CT args ---
    reader_rt  = [[input_tensor.buffer_address(),  num_tiles]]
    writer_rt  = [[output_tensor.buffer_address(), num_tiles]]
    compute_ct = [num_tiles]  # per_core_tile_cnt as CT arg (index 0)
    compute_rt = []

    # --- Kernel descriptors ---
    reader_k = ttnn.KernelDescriptor(
        kernel_source=read_tiles_src,
        source_type=ttnn._ttnn.program_descriptor.SourceType.SOURCE_CODE,
        core_ranges=grid,
        compile_time_args=reader_ct,
        runtime_args=[reader_rt],
        config=ttnn.ReaderConfigDescriptor(),
    )
    compute_k = ttnn.KernelDescriptor(
        kernel_source=compute_src,
        source_type=ttnn._ttnn.program_descriptor.SourceType.SOURCE_CODE,
        core_ranges=grid,
        compile_time_args=compute_ct,
        runtime_args=[compute_rt],
        config=ttnn.ComputeConfigDescriptor(),
    )
    writer_k = ttnn.KernelDescriptor(
        kernel_source=write_tiles_src,
        source_type=ttnn._ttnn.program_descriptor.SourceType.SOURCE_CODE,
        core_ranges=grid,
        compile_time_args=writer_ct,
        runtime_args=[writer_rt],
        config=ttnn.WriterConfigDescriptor(),
    )

    prog = ttnn.ProgramDescriptor(
        kernels=[reader_k, compute_k, writer_k],
        semaphores=[],
        cbs=[in_cb_desc, out_cb_desc],
    )

    return ttnn.generic_op([input_tensor, output_tensor], prog)


def get_inputs(case: int):
    B = D = 32
    if case == 0:
        B = D = 1
    elif case == 1:
        B = 1
        D = 2
    elif case == 2:
        B = 2
        D = 1
    elif case == 3:
        B = 2
        D = 2
    elif case == 4:
        B = D = 64

    return (B, D)

def main():
    dev = ttnn.open_device(device_id=0)
    case = 4
    size = get_inputs(case=case)

    # keep a safe margin above 1 to avoid extreme conditioning in BF16
    x = torch.rand((size), dtype=torch.bfloat16)  # ∈ [1, 1.5)
    x_tt = ttnn.from_torch(x, device=dev, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    y_tt = host(x_tt)            # your acosh pipeline
    y = ttnn.to_torch(y_tt, device=dev)

    ref = torch.sqrt(x).to(torch.bfloat16)  # BF16 match
    print("max_err:", torch.max(torch.abs(y - ref)))
    print("allclose:", torch.allclose(y, ref, rtol=1e-2, atol=1e-2))

if __name__ == "__main__":
    main()