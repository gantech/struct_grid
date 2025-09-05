############################ BASE ######################################
cmake_minimum_required (VERSION 3.17.0 FATAL_ERROR)

# Set the project name
project(StructGridCUDA LANGUAGES CXX CUDA)


# Set the CUDA version (if not automatically detected)
find_package(CUDA 12.0 REQUIRED)
set(CUDA_COMPUTE_CAPABILITY 90)
set(CMAKE_CXX_STANDARD 20)

add_subdirectory(src)
