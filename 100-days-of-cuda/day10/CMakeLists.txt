cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(tile_matmul LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Set C++ standard (optional but recommended)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Define the executable from your .cu file
add_executable(run_matmul main.cu)

# This makes sure -fPIC and similar flags are passed properly,
# which is needed by some Torch builds
set_property(TARGET run_matmul PROPERTY POSITION_INDEPENDENT_CODE ON)
