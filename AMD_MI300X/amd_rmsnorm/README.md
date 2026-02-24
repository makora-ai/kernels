# Different RMSNorm implementations for AMD MI300X

Optional dependencies include liger and apex.
Please install them by following their respective READMEs.

The repo also provides a standalone HIP implementation based on Apex kernels - please run the `compile_apex.py` script to compile it.
The compilation process uses PyTorch's extension mechanism to compile the source code, therefore please make sure a correct PyTorch installation is present.

Afterwards `test.py` can be used to test correctness of different implementations.
Example output should look like the following:

```
Shape            Type            Impl.                Fwd.    Bwd.
---------------  --------------  -------------------  ------  ------
(1, 4096, 8192)  torch.float32   apex                 OK      OK
(1, 4096, 8192)  torch.float32   custom-apex          OK      OK
(1, 4096, 8192)  torch.float32   liger                OK      OK
(1, 4096, 8192)  torch.float32   liger-inplace        OK      OK
(1, 4096, 8192)  torch.float32   orig-llama           OK      OK
(1, 4096, 8192)  torch.float32   orig-llama-compiled  OK      OK
(1, 4096, 8192)  torch.float32   pytorch-compiled     OK      OK

(1, 4096, 8192)  torch.float16   apex                 OK      OK
(1, 4096, 8192)  torch.float16   custom-apex          OK      OK
(1, 4096, 8192)  torch.float16   liger                OK      OK
(1, 4096, 8192)  torch.float16   liger-inplace        OK      OK
(1, 4096, 8192)  torch.float16   orig-llama           OK      OK
(1, 4096, 8192)  torch.float16   orig-llama-compiled  OK      OK
(1, 4096, 8192)  torch.float16   pytorch-compiled     OK      OK

(1, 4096, 8192)  torch.bfloat16  apex                 OK      OK
(1, 4096, 8192)  torch.bfloat16  custom-apex          OK      OK
(1, 4096, 8192)  torch.bfloat16  liger                OK      OK
(1, 4096, 8192)  torch.bfloat16  liger-inplace        OK      OK
(1, 4096, 8192)  torch.bfloat16  orig-llama           OK      OK
(1, 4096, 8192)  torch.bfloat16  orig-llama-compiled  OK      OK
(1, 4096, 8192)  torch.bfloat16  pytorch-compiled     OK      OK
```

If functional tests are all ok, benchmarking can then be performed by running `benchmark.py`.

```
Shape            Type            Impl.                Inf.             Fwd.             Bwd.
---------------  --------------  -------------------  ---------------  ---------------  ---------------
(1, 4096, 8192)  torch.float32   apex                 0.1188 ± 0.0144  0.0985 ± 0.0017  0.4184 ± 0.0024
(1, 4096, 8192)  torch.float32   custom-apex          0.1108 ± 0.0018  0.0994 ± 0.0016  0.4201 ± 0.0022
(1, 4096, 8192)  torch.float32   liger                0.0943 ± 0.0136  0.0882 ± 0.0117  0.2266 ± 0.0019
(1, 4096, 8192)  torch.float32   liger-inplace        0.0913 ± 0.0117  0.0817 ± 0.0122  0.2068 ± 0.0053
(1, 4096, 8192)  torch.float32   orig-llama           0.2770 ± 0.0134  0.2816 ± 0.0117  1.1714 ± 0.0181
(1, 4096, 8192)  torch.float32   orig-llama-compiled  0.1022 ± 0.0143  0.0820 ± 0.0040  0.2901 ± 0.0102
(1, 4096, 8192)  torch.float32   pytorch              0.2826 ± 0.0108  0.2914 ± 0.0084  1.2305 ± 0.0267
(1, 4096, 8192)  torch.float32   pytorch-compiled     0.0992 ± 0.0144  0.0818 ± 0.0031  0.2884 ± 0.0042

(1, 4096, 8192)  torch.float16   apex                 0.0501 ± 0.0009  0.0502 ± 0.0009  0.2050 ± 0.0025
(1, 4096, 8192)  torch.float16   custom-apex          0.0560 ± 0.0099  0.0545 ± 0.0059  0.2060 ± 0.0048
(1, 4096, 8192)  torch.float16   liger                0.0781 ± 0.0051  0.0662 ± 0.0163  0.1431 ± 0.0072
(1, 4096, 8192)  torch.float16   liger-inplace        0.0792 ± 0.0069  0.0729 ± 0.0112  0.1265 ± 0.0016
(1, 4096, 8192)  torch.float16   orig-llama           0.3921 ± 0.0328  0.3944 ± 0.0182  1.1849 ± 0.0412
(1, 4096, 8192)  torch.float16   orig-llama-compiled  0.0540 ± 0.0063  0.0536 ± 0.0084  0.1838 ± 0.0135
(1, 4096, 8192)  torch.float16   pytorch              0.1887 ± 0.0116  0.1834 ± 0.0085  0.7778 ± 0.0244
(1, 4096, 8192)  torch.float16   pytorch-compiled     0.0548 ± 0.0079  0.0535 ± 0.0064  0.1879 ± 0.0132

(1, 4096, 8192)  torch.bfloat16  apex                 0.0589 ± 0.0015  0.0595 ± 0.0011  0.2099 ± 0.0051
(1, 4096, 8192)  torch.bfloat16  custom-apex          0.0704 ± 0.0099  0.0677 ± 0.0105  0.2107 ± 0.0054
(1, 4096, 8192)  torch.bfloat16  liger                0.0603 ± 0.0075  0.0560 ± 0.0032  0.1634 ± 0.0025
(1, 4096, 8192)  torch.bfloat16  liger-inplace        0.0582 ± 0.0026  0.0565 ± 0.0028  0.1486 ± 0.0007
(1, 4096, 8192)  torch.bfloat16  orig-llama           0.3918 ± 0.0171  0.3954 ± 0.0149  1.1927 ± 0.0284
(1, 4096, 8192)  torch.bfloat16  orig-llama-compiled  0.0517 ± 0.0017  0.0537 ± 0.0073  0.1802 ± 0.0121
(1, 4096, 8192)  torch.bfloat16  pytorch              0.1887 ± 0.0053  0.1859 ± 0.0053  0.8111 ± 0.0282
(1, 4096, 8192)  torch.bfloat16  pytorch-compiled     0.0547 ± 0.0062  0.0527 ± 0.0056  0.1823 ± 0.0120
```

When benchmarking, `Inf.` refers to performing a forward pass within `torch.inference_mode()` context, while `Fwd.` measures forward time for the purpose of the following backward.
