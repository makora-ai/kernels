import torch
import ttnn
import ttnn._ttnn
from pathlib import Path

def load_file(path: Path) -> str:
    return path.read_text()

# Assume current working directory is the project root: tt-dataset/
ROOT = Path.cwd()
EXAMPLES_DIR = ROOT / "kernels" / "unary" / "trigonometric" / "atanh"

READ_SRC_PATH    = EXAMPLES_DIR / "read.cpp"
WRITE_SRC_PATH   = EXAMPLES_DIR / "write.cpp"
COMPUTE_SRC_PATH = EXAMPLES_DIR / "compute.cpp"

read_tiles_src = load_file(READ_SRC_PATH)
write_tiles_src = load_file(WRITE_SRC_PATH)
compute_src = load_file(COMPUTE_SRC_PATH)


def host(input_tensor: ttnn.Tensor) -> ttnn.Tensor:
    """
    Runs elementwise atanh on input_tensor using a 3-kernel pipeline.
    Compute kernel expects per_core_tile_cnt as a COMPILE-TIME arg (index 0).
    """
    # Output tensor (same shape/dtype/layout/device)
    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(input_tensor.shape),
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        input_tensor.device(),
    )

    # ---- CB config ----
    tiles_per_cb = 2
    tile_bytes = 32 * 32 * 2  # bf16
    cb_total_size = tiles_per_cb * tile_bytes
    cb_page_size = tile_bytes

    # CB indices must match the kernels
    in_cb, out_cb = 0, 16

    in_cb_format = ttnn.CBFormatDescriptor(
        buffer_index=in_cb, data_format=ttnn.bfloat16, page_size=cb_page_size
    )
    out_cb_format = ttnn.CBFormatDescriptor(
        buffer_index=out_cb, data_format=ttnn.bfloat16, page_size=cb_page_size
    )

    core = ttnn.CoreCoord(0, 0)
    core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])

    in_cb_desc = ttnn.CBDescriptor(
        total_size=cb_total_size, core_ranges=core_grid, format_descriptors=[in_cb_format]
    )
    out_cb_desc = ttnn.CBDescriptor(
        total_size=cb_total_size, core_ranges=core_grid, format_descriptors=[out_cb_format]
    )

    # ---- Accessor CT args ----
    reader_ct_args = ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args()
    writer_ct_args = ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args()

    # ---- Tiles count (FIXED: use input shape, not H/W) ----
    B, D = input_tensor.shape

    Mt = B // 32
    Nt = D // 32
    num_tiles = max(1, Mt * Nt)

    # ---- RT/CT args ----
    reader_rt_args = [[input_tensor.buffer_address(), num_tiles]]
    writer_rt_args = [[output_tensor.buffer_address(), num_tiles]]

    # Compute expects CT arg 0 = per_core_tile_cnt
    compute_ct_args = [num_tiles]
    compute_rt_args = []  # none

    # ---- Kernel descriptors ----
    reader_k = ttnn.KernelDescriptor(
        kernel_source=read_tiles_src,
        source_type=ttnn._ttnn.program_descriptor.SourceType.SOURCE_CODE,
        core_ranges=core_grid,
        compile_time_args=reader_ct_args,
        runtime_args=[reader_rt_args],
        config=ttnn.ReaderConfigDescriptor(),
    )

    compute_k = ttnn.KernelDescriptor(
        kernel_source=compute_src,
        source_type=ttnn._ttnn.program_descriptor.SourceType.SOURCE_CODE,
        core_ranges=core_grid,
        compile_time_args=compute_ct_args,   # per_core_tile_cnt
        runtime_args=[compute_rt_args],
        config=ttnn.ComputeConfigDescriptor(),
    )

    writer_k = ttnn.KernelDescriptor(
        kernel_source=write_tiles_src,
        source_type=ttnn._ttnn.program_descriptor.SourceType.SOURCE_CODE,
        core_ranges=core_grid,
        compile_time_args=writer_ct_args,
        runtime_args=[writer_rt_args],
        config=ttnn.WriterConfigDescriptor(),
    )

    program_descriptor = ttnn.ProgramDescriptor(
        kernels=[reader_k, compute_k, writer_k],
        semaphores=[],
        cbs=[in_cb_desc, out_cb_desc],
    )

    return ttnn.generic_op([input_tensor, output_tensor], program_descriptor)


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


    return (B, D)


def main():
    dev = ttnn.open_device(device_id=0)
    case = 2
    size = get_inputs(case=case)
    x = torch.rand(size, dtype=torch.bfloat16)  # 2×2 tiles
    x_tt = ttnn.from_torch(x, device=dev, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    y_tt = host(x_tt)
    y = ttnn.to_torch(y_tt, device=dev)
    ref = torch.atanh(x)
    print("allclose:", torch.allclose(y, ref, rtol=1e-2, atol=1e-2))

if __name__ == "__main__":
    main()
