
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


cuda_add_library(linear_solvers STATIC LinearSolver.cu Jacobi.cu ADI.cu CG.cu Multigrid.cu)
set_target_properties(linear_solvers PROPERTIES CUDA_ARCHITECTURES "90")
install(TARGETS linear_solvers DESTINATION lib)

cuda_add_executable(laplace_heat_jacobi LaplaceHeat.cu)
set_target_properties(laplace_heat_jacobi PROPERTIES CUDA_ARCHITECTURES "90")
target_link_libraries(laplace_heat_jacobi linear_solvers)
install(TARGETS laplace_heat_jacobi DESTINATION bin)

cuda_add_executable(lid_driven_cavity LidDrivenCavity.cu)
set_target_properties(lid_driven_cavity PROPERTIES CUDA_ARCHITECTURES "90")
target_link_libraries(lid_driven_cavity linear_solvers)
install(TARGETS lid_driven_cavity DESTINATION bin)