from pathlib import Path

import os

import torch.utils.cpp_extension
# Monkey-patch to bypass CUDA version check
torch.utils.cpp_extension._check_cuda_version = lambda *args, **kwargs: None

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension


def find_cpp_files(directories):
    out = []
    for dir in directories:
        dir = Path(dir)
        out.extend(dir.rglob('*.cpp'))
        out.extend(dir.rglob('*.cu'))

    return out


cpp_files = find_cpp_files(['bindings', 'kernels', 'pbr_matrix'])
include_dirs = [os.path.join(os.getcwd())]

compile_cuda = bool(os.environ.get("CUDA_ENABLED", True))
if compile_cuda:
    print("Compiling with CUDA support.")
    os.environ["TORCH_CUDA_ARCH_LIST"] = "8.6"
    # Update this section in your setup.py:
    extension = CUDAExtension(
        name='graph_link_core',
        sources=cpp_files,
        include_dirs=include_dirs + ['ext/cuCollections/include'],
        extra_compile_args={
            'nvcc': [
                '-std=c++17', # Downgrade to 17 for better compatibility
                '-lineinfo', 
                '-O3', 
                '-allow-unsupported-compiler'
            ],
            'cxx': [
                '-DCUDA_ENABLED=1', 
                '-O3', 
                '-std=c++17', # Downgrade here as well
                '-w'
            ]
        }
    )
else:
    print("Compiling without CUDA support.")
    cpp_files = [f for f in cpp_files if not f.suffix == '.cu']
    extension = CppExtension(name='graph_link_core',
                             sources=cpp_files,
                             include_dirs=include_dirs,
                             extra_compile_args={'cxx': ['-O3', '-std=c++20', '-w'], })

setup(name='graph_link_core',
      version='0.0.1',
      ext_modules=[extension],
      cmdclass={'build_ext': BuildExtension},
      packages=['.'])
