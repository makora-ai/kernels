# SPDX-License-Identifier: Apache-2.0
import struct
from pathlib import Path
import torch
import ttnn
import ttnn._ttnn

DTYPE = ttnn.bfloat16

def load_file(path: str) -> str:
    with open(path, "r") as f:
        return f.read()

def f32_to_u32_bits(f: float) -> int:
    # Convert Python float -> IEEE-754 float32 -> uint32 bit pattern
    return struct.unpack("<I", struct.pack("<f", float(f)))[0]

ROOT = Path.cwd()
EXAMPLES_DIR = ROOT / "kernels" / "unary" / "specialized_functions" / "celu"

READ_SRC_PATH    = EXAMPLES_DIR / "read.cpp"
WRITE_SRC_PATH   = EXAMPLES_DIR / "write.cpp"
COMPUTE_SRC_PATH = EXAMPLES_DIR / "compute.cpp"

read_tiles_src  = load_file(READ_SRC_PATH)
write_tiles_src = load_file(WRITE_SRC_PATH)
compute_src     = load_file(COMPUTE_SRC_PATH)

def host(x: ttnn.Tensor, alpha: float = 1.0) -> ttnn.Tensor:
    """
    CELU on TT via 3-kernel pipeline.
      - Reader RT:  [src_base, num_tiles]
      - Compute CT: [per_core_tile_cnt=num_tiles]
      - Compute RT: [alpha_bits, alpha_recip_bits]
      - Writer RT:  [dst_base, num_tiles]
    PyTorch ref: torch.nn.functional.celu(x, alpha)
    """
    y = ttnn.allocate_tensor_on_device(
        ttnn.Shape(x.shape), DTYPE, ttnn.TILE_LAYOUT, x.device()
    )

    # --- CB config ---
    tile_bytes   = 32 * 32 * 2  # 2048B BF16 tile
    tiles_per_cb = 2
    total_bytes  = tiles_per_cb * tile_bytes
    cb_in, cb_out = 0, 16

    in_fmt  = ttnn.CBFormatDescriptor(buffer_index=cb_in,  data_format=DTYPE, page_size=tile_bytes)
    out_fmt = ttnn.CBFormatDescriptor(buffer_index=cb_out, data_format=DTYPE, page_size=tile_bytes)

    core = ttnn.CoreCoord(0, 0)
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])

    in_cb  = ttnn.CBDescriptor(total_size=total_bytes, core_ranges=grid, format_descriptors=[in_fmt])
    out_cb = ttnn.CBDescriptor(total_size=total_bytes, core_ranges=grid, format_descriptors=[out_fmt])

    # --- Tiles count ---
    B, D = x.shape
    Mt = B // 32
    Nt = D // 32
    num_tiles = max(1, Mt * Nt)

    # --- Accessor CT args ---
    reader_ct = ttnn.TensorAccessorArgs(x).get_compile_time_args()
    writer_ct = ttnn.TensorAccessorArgs(y).get_compile_time_args()

    # --- Runtime args ---
    reader_rt = [[x.buffer_address(), num_tiles]]
    writer_rt = [[y.buffer_address(), num_tiles]]

    # Compute: CT per-core count, RT alpha params as bit patterns
    alpha_bits       = f32_to_u32_bits(alpha)
    alpha_recip_bits = f32_to_u32_bits(1.0 / alpha)

    compute_ct = [num_tiles]
    compute_rt = [[alpha_bits, alpha_recip_bits]]

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


# Tiny check vs PyTorch reference
def main():
    """
    Pytorch reference: https://docs.pytorch.org/docs/stable/generated/torch.nn.CELU.html
    """
    dev = ttnn.open_device(device_id=0)
    case = 5
    size = get_inputs(case=case)
    X = (torch.rand(size) - 0.5) * 6
    X = X.to(torch.bfloat16)

    Xtt = ttnn.from_torch(X, device=dev, dtype=DTYPE, layout=ttnn.TILE_LAYOUT)

    alpha = 1 # 1.1
    Ytt = host(Xtt, alpha=alpha)
    Y = ttnn.to_torch(Ytt, device=dev)

    ref = torch.nn.CELU(alpha=alpha)(X.to(torch.float32)).to(torch.bfloat16)
    print("max_err:", torch.max(torch.abs(Y - ref)))
    print("allclose:", torch.allclose(Y, ref, rtol=1e-2, atol=1e-2))
    print(ref)
    print(Y)

if __name__ == "__main__":
    main()
