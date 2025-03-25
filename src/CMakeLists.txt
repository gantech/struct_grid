
# # ADI in CUDA
# add_executable(adi_cuda try_adi.cu laplace_heat.cu adi.cu)

# #Compile specifically for H100's on Kestrel
# set_target_properties(adi_cuda PROPERTIES
#     CUDA_ARCHITECTURES "90"
# )
# install(TARGETS adi_cuda DESTINATION bin)

# # Multigrid in CUDA
# add_executable(adi_mg multigrid.cu laplace_heat.cu adi.cu)
# set_target_properties(adi_mg PROPERTIES CUDA_ARCHITECTURES "90")
# install(TARGETS adi_mg DESTINATION bin)


# add_subdirectory(airfoil)


cuda_add_library(linear_solvers STATIC LinearSolvers.cpp Jacobi.cpp jacobi.cu)
set_target_properties(linear_solvers PROPERTIES CUDA_ARCHITECTURES "90")
# Force nvcc for the .cpp file
set_target_properties(linear_solvers PROPERTIES CUDA_SOURCE_FILE_EXTENSIONS ".cpp"
)
install(TARGETS linear_solvers DESTINATION lib)
