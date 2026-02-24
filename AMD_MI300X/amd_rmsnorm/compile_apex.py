import os

from torch.utils.cpp_extension import load

build_dir = os.path.join(os.path.dirname(__file__), 'implementations/custom_apex')

cwd = os.getcwd()
os.chdir(build_dir)
try:
    apex_C = load(name="C", sources=[
            "csrc/forward.cu",
            "csrc/backward.cu",
            "csrc/kernel.cpp"
        ],
        extra_cflags=['-O3', '-std=c++17'],
        extra_cuda_cflags=['-O3', '--offload-arch=gfx942', '-std=c++17'],
        with_cuda=True,
        build_directory=build_dir,
        keep_intermediates=False)
finally:
    os.chdir(cwd)

for file in os.listdir(build_dir):
    if file.endswith('.o') or file in ['build.ninja', '.ninja_deps', '.ninja_log']:
        os.unlink(os.path.join(build_dir, file))
