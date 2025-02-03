from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='vec_softmax_cuda',
    ext_modules=[
        CUDAExtension('vec_softmax_cuda',
                      sources=[
                          'vec_softmax.cu',
                          'vec_softmax_binding.cpp',
                      ],
                      ),
    ],
    cmdclass={'build_ext': BuildExtension}
)
