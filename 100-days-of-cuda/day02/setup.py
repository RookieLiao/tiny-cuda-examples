from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='warp_softmax_cuda',
    ext_modules=[
        CUDAExtension('warp_softmax_cuda',
                      sources=[
                          'warp_softmax.cu',
                          'warp_softmax_binding.cpp',
                      ],
                      extra_compile_args={
                          'nvcc': ['-O3', '-std=c++17'],
                      },
                      ),
    ],
    cmdclass={'build_ext': BuildExtension}
)
