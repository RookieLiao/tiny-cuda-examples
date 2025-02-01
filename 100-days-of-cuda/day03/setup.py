from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='dtype_aware_softmax_cuda',
    ext_modules=[
        CUDAExtension('dtype_aware_softmax_cuda', [
            'dtype_aware_softmax.cu',
            'dtype_aware_softmax_binding.cpp',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
