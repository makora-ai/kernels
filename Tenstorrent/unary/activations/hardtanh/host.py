# SPDX-License-Identifier: Apache-2.0
import struct
import torch
import ttnn
import ttnn._ttnn
from pathlib import Path

def load_file(path: str) -> str:
    with open(path, "r") as f:
        return f.read()

def f32_to_u32_bits(f: float) -> int:
    # Python float -> IEEE-754 float32 -> uint32 bit pattern
    return struct.unpack("<I", struct.pack("<f", float(f)))[0]

ROOT = Path.cwd()
EXAMPLES_DIR = ROOT / "kernels" / "unary" / "activations" / "hardtanh"  # <-- hardtanh folder

READ_SRC_PATH    = EXAMPLES_DIR / "read.cpp"
WRITE_SRC_PATH   = EXAMPLES_DIR / "write.cpp"
COMPUTE_SRC_PATH = EXAMPLES_DIR / "compute.cpp"

read_tiles_src  = load_file(READ_SRC_PATH)
write_tiles_src = load_file(WRITE_SRC_PATH)
compute_src     = load_file(COMPUTE_SRC_PATH)

def host(x: ttnn.Tensor, min_val: float = -1.0, max_val: float = 1.0) -> ttnn.Tensor:
    """
    HardTanh on TT via 3-kernel pipeline.
    - Reader RT: [src_base, num_tiles]
    - Compute CT: [per_core_tile_cnt=num_tiles]
    - Compute RT: [min_bits, max_bits]  (float32 bit patterns)
    - Writer RT: [dst_base, num_tiles]
    PyTorch reference: y = clamp(x, min_val, max_val)
    """
    # Output mirrors input
    y = ttnn.allocate_tensor_on_device(
        ttnn.Shape(x.shape), ttnn.bfloat16, ttnn.TILE_LAYOUT, x.device()
    )

    # --- CB config ---
    tile_bytes   = 32 * 32 * 2  # BF16
    tiles_per_cb = 2
    total_bytes  = tiles_per_cb * tile_bytes
    cb_in, cb_out = 0, 16

    in_fmt  = ttnn.CBFormatDescriptor(buffer_index=cb_in,  data_format=ttnn.bfloat16, page_size=tile_bytes)
    out_fmt = ttnn.CBFormatDescriptor(buffer_index=cb_out, data_format=ttnn.bfloat16, page_size=tile_bytes)

    core = ttnn.CoreCoord(0, 0)
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])

    in_cb  = ttnn.CBDescriptor(total_size=total_bytes, core_ranges=grid, format_descriptors=[in_fmt])
    out_cb = ttnn.CBDescriptor(total_size=total_bytes, core_ranges=grid, format_descriptors=[out_fmt])

    # --- Tiles count ---
    B, D = x.shape
    Mt = B // 32
    Nt = D // 32
    num_tiles = max(1, Mt * Nt)

    # --- CT/RT args ---
    reader_ct = ttnn.TensorAccessorArgs(x).get_compile_time_args()
    writer_ct = ttnn.TensorAccessorArgs(y).get_compile_time_args()

    reader_rt = [[x.buffer_address(), num_tiles]]
    writer_rt = [[y.buffer_address(), num_tiles]]

    compute_ct = [num_tiles]  # per_core_tile_cnt

    # Pass min/max as float32 bit-patterns to compute (RT)
    min_bits = f32_to_u32_bits(min_val)
    max_bits = f32_to_u32_bits(max_val)
    compute_rt = [[min_bits, max_bits]]

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
        cbs=[in_cb, out_cb],
    )

    return ttnn.generic_op([x, y], prog)

def get_inputs(case: int):
    B = D = 64
    if case == 0:
        B = D = 1
    elif case == 1:
        B = 1; D = 2
    elif case == 2:
        B = 2; D = 1
    elif case == 3:
        B = 2; D = 2
    elif case == 4:
        B = D = 32
    return (B, D)

# Tiny check vs PyTorch reference: clamp(x, min_val, max_val)
def main():
    torch.manual_seed(0)
    dev = ttnn.open_device(device_id=0)
    case = 5
    size = get_inputs(case=case)

    X = (torch.rand(size) - 0.5) * 8  # wider range to exercise clamping
    X = X.to(torch.bfloat16)

    Xtt = ttnn.from_torch(X, device=dev, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    min_val, max_val = -1.0, 1.0
    Ytt = host(Xtt, min_val=min_val, max_val=max_val)
    Y = ttnn.to_torch(Ytt, device=dev)

    ref = torch.clamp(X.to(torch.float32), min=min_val, max=max_val).to(torch.bfloat16)
    print("max_err:", torch.max(torch.abs(Y - ref)))
    print("allclose:", torch.allclose(Y, ref, rtol=1e-2, atol=1e-2))

if __name__ == "__main__":
    main()
