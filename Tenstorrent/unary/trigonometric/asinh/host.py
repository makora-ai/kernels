# SPDX-License-Identifier: Apache-2.0
import torch
import ttnn
import ttnn._ttnn
from pathlib import Path

def load_file(path: str) -> str:
    with open(path, "r") as f:
        return f.read()

ROOT = Path.cwd()
EXAMPLES_DIR = ROOT / "kernels" / "unary" / "trigonometric" / "asinh"

READ_SRC_PATH    = EXAMPLES_DIR / "read.cpp"
WRITE_SRC_PATH   = EXAMPLES_DIR / "write.cpp"
COMPUTE_SRC_PATH = EXAMPLES_DIR / "compute.cpp"

read_src  = load_file(READ_SRC_PATH)
write_src = load_file(WRITE_SRC_PATH)
comp_src  = load_file(COMPUTE_SRC_PATH)

def host(x: ttnn.Tensor) -> ttnn.Tensor:
    # Output mirrors input
    y = ttnn.allocate_tensor_on_device(
        ttnn.Shape(x.shape), ttnn.bfloat16, ttnn.TILE_LAYOUT, x.device()
    )

    # CBs
    tile_bytes   = 32*32*2
    tiles_per_cb = 2
    total_bytes  = tiles_per_cb * tile_bytes
    cb_in, cb_out = 0, 16

    in_fmt  = ttnn.CBFormatDescriptor(buffer_index=cb_in,  data_format=ttnn.bfloat16, page_size=tile_bytes)
    out_fmt = ttnn.CBFormatDescriptor(buffer_index=cb_out, data_format=ttnn.bfloat16, page_size=tile_bytes)

    core = ttnn.CoreCoord(0, 0)
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])

    in_cb  = ttnn.CBDescriptor(total_size=total_bytes, core_ranges=grid, format_descriptors=[in_fmt])
    out_cb = ttnn.CBDescriptor(total_size=total_bytes, core_ranges=grid, format_descriptors=[out_fmt])

    # --- Tile count ---
    B, D = x.shape

    # tiles count
    Mt = B // 32
    Nt = D // 32
    num_tiles = max(1, Mt * Nt)

    # CT/RT args
    reader_ct = ttnn.TensorAccessorArgs(x).get_compile_time_args()
    writer_ct = ttnn.TensorAccessorArgs(y).get_compile_time_args()
    reader_rt = [[x.buffer_address(),  num_tiles]]
    writer_rt = [[y.buffer_address(),  num_tiles]]
    compute_ct = [num_tiles]  # per_core_tile_cnt
    compute_rt = []

    reader_k = ttnn.KernelDescriptor(
        kernel_source=read_src,
        source_type=ttnn._ttnn.program_descriptor.SourceType.SOURCE_CODE,
        core_ranges=grid,
        compile_time_args=reader_ct,
        runtime_args=[reader_rt],
        config=ttnn.ReaderConfigDescriptor(),
    )
    compute_k = ttnn.KernelDescriptor(
        kernel_source=comp_src,
        source_type=ttnn._ttnn.program_descriptor.SourceType.SOURCE_CODE,
        core_ranges=grid,
        compile_time_args=compute_ct,
        runtime_args=[compute_rt],
        config=ttnn.ComputeConfigDescriptor(),
    )
    writer_k = ttnn.KernelDescriptor(
        kernel_source=write_src,
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
    case = 5
    size = get_inputs(case=case)

    # keep a safe margin above 1 to avoid extreme conditioning in BF16
    x = 1.0 + 0.5 * torch.rand(64, 64, dtype=torch.bfloat16)  # ∈ [1, 1.5)
    x_tt = ttnn.from_torch(x, device=dev, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    y_tt = host(x_tt)            # your acosh pipeline
    y = ttnn.to_torch(y_tt, device=dev)

    ref = torch.asinh(x).to(torch.bfloat16)  # BF16 match
    print("max_err:", torch.max(torch.abs(y - ref)))
    print("allclose:", torch.allclose(y, ref, rtol=1e-2, atol=1e-2))


if __name__ == "__main__":
    main()
