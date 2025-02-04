from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="matmul_cuda",
    ext_modules=[
        CUDAExtension(
            "matmul_cuda",
            sources=[
                "tile_matmul.cu",
                "matmul_binding.cpp",
            ],
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
