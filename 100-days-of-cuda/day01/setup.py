from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='softmax_cuda',
    ext_modules=[
        CUDAExtension('softmax_cuda',
                      sources=[
            'softmax.cu',
            'softmax_binding.cpp',
        ],
        libraries=['torch'],
        extra_compile_args={
            'nvcc': ['-O3', '-std=c++17'],
        },
        ),
    ],
    cmdclass={'build_ext': BuildExtension}
)
