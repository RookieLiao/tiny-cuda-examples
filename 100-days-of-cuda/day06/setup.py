from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="naive_matmul_cuda",
    ext_modules=[
        CUDAExtension(
            "naive_matmul_cuda",
            sources=[
                "matmul.cu",
                "matmul_binding.cpp",
            ],
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
