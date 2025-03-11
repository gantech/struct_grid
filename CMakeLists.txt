############################ BASE ######################################
cmake_minimum_required (VERSION 3.17.0 FATAL_ERROR)
project(struct_grid CXX)

# Set the project name
project(StructGridCUDA LANGUAGES CXX CUDA)


# Set the CUDA version (if not automatically detected)
find_package(CUDA 12.0 REQUIRED)

# Set CUDA architecture (optional)
# If you're targeting specific CUDA architectures, you can set:
set(CUDA_ARCHITECTURES "90")

add_subdirectory(src)
