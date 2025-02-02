from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='block_softmax_cuda',
    ext_modules=[
        CUDAExtension('block_softmax_cuda',
            sources=[
                'block_softmax.cu',
                'block_softmax_binding.cpp',
            ],
            extra_compile_args={'nvcc': ['-O3']}
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)

