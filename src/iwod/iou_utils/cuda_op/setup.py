from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Compute Capability A100 is 8.0
cuda_arch_flags = [
    '-gencode=arch=compute_80,code=sm_80'
]

setup(
    name='sort_vertices',
    ext_modules=[
        CUDAExtension(
            'sort_vertices',
            sources=[
                'sort_vert.cpp',
                'sort_vert_kernel.cu',
            ],
            extra_compile_args={
                'cxx': [],
                'nvcc': cuda_arch_flags
            }
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)