# Tenstorrent Kernels

This directory contains kernel implementations generated automatically using **MakoraGenerate** and validated on Tenstorrent hardware. Kernels are implemented using Tenstorrent’s supported toolchain and programming model (as specified per kernel).

# Kernel Benchmark Results

| Op | Category | Device Speedup (GMean) | Total Speedup | Per-Shape Speedups |
|----|----------|------------------------|---------------|--------------------|
| atan2 | binary | **5.12x** | **2.65x** | [32,32] → **6.22x**<br>[4,384] → **6.60x**<br>[4,384,4096] → **4.09x**<br>[4,1,384,4096] → **4.09x** |
| isclose | binary | **5.05x** | **2.00x** | [32,32] → **5.46x**<br>[4,384] → **5.81x**<br>[4,384,4096] → **4.53x**<br>[4,1,384,4096] → **4.53x** |
| nextafter | binary | **5.46x** | **1.47x** | [32,32] → **5.13x**<br>[4,384] → **5.21x**<br>[4,384,4096] → **5.77x**<br>[4,1,384,4096] → **5.77x** |
| outer | binary | **2.88x** | **0.55x** | [1,1,32,1] x [1,1,1,32] → **2.54x**<br>[1,1,64,1] x [1,1,1,128] → **2.39x**<br>[32,1,1,1] x [1,32,1,1] → **3.92x** |
| remainder | binary | **3.32x** | **1.55x** | [32,32] → **2.99x**<br>[4,384] → **2.99x**<br>[4,384,4096] → **3.69x**<br>[4,1,384,4096] → **3.69x** |
| digamma | unary | **6.26x** | **2.56x** | [32,128] → **7.35x**<br>[5,2240,32] → **6.90x**<br>[3,2,32,5600] → **4.83x** |
| glu | unary | **2.79x** | **0.91x** | [32,32,32,64] → **3.02x**<br>[3,2,32,4096] → **2.58x** |
| lgamma | unary | **3.80x** | **3.03x** | [32,32] → **4.30x**<br>[32,128] → **4.40x**<br>[5,2240,32] → **2.89x** |
| multigammaln | unary | **5.06x** | **7.21x** | [32,32] → **5.79x**<br>[32,128] → **5.93x**<br>[5,2240,32] → **3.78x** |
| polygamma | unary | **2.80x** | **2.89x** | [32,32] → **3.41x**<br>[32,128] → **3.60x**<br>[5,2240,32] → **2.65x**<br>[3,2,32,5600] → **1.89x** |
| reglu | unary | **2.93x** | **1.15x** | [1,1,32,64] → **2.69x**<br>[1,1,128,512] → **3.00x**<br>[3,2,1024,4096] → **3.10x** |
| swiglu | unary | **2.31x** | **1.00x** | [1,1,32,64] → **1.98x**<br>[1,1,128,512] → **2.26x**<br>[1,1,1024,4096] → **2.76x** |
| triu | unary | **0.57x** | **5.09x** | [32,32] → **1.14x**<br>[32,64] → **1.14x**<br>[4,384,4096] → **0.28x**<br>[4,1,384,4096] → **0.28x** |
