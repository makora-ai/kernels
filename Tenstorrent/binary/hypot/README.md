## Hypot tt-metal Kernel Generation

The directory contains the trajectory of generating the `hypot` kernel. 

Using the implementation [here](https://github.com/tenstorrent/tt-metal/blob/9b8011f3ca7aa11b4c56cf37dc2b4cac5a205b88/ttnn/cpp/ttnn/operations/eltwise/binary/device/binary_composite_op.cpp#L25). 

The tt-metal `hypot` kernel uses the following implementation

```
Tensor _hypot(const Tensor& input_a, const Tensor& input_b, const std::optional<MemoryConfig>& output_mem_config) {
    Tensor a_sq = ttnn::square(input_a, output_mem_config);
    Tensor b_sq = ttnn::square(input_b, output_mem_config);
    Tensor c_sq = ttnn::add(a_sq, b_sq, std::nullopt, output_mem_config);
    a_sq.deallocate();
    b_sq.deallocate();
    return ttnn::sqrt(c_sq, output_mem_config);
}

```

In simple terms it is `sqrt(x^2 + y^2)`, broken down by using `ttnn::square`, `ttnn::add`, `ttnn::sqrt`.

The `hypot` kernel was generated using the above as building blocks and fusing them incrementally. 

There are three `hypot` kernel implemetation. 
1. Building the tt-metal kernels using the tt-llk intrinsics, running each host code sequentially. Three host codes, three compute kernels. Saved in [sequential](./sequential)
2. Fusing the add and square. Two host codes, two compute kernels, running each host code sequentially. Saved in [fused_square_add](./fused_square_add)
3. Fusing all three. One host code and one compute kernel. Saved in [fused](./fused)

