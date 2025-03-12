# Green-Gauss gradient on airfoil
add_executable(airfoil_grad airfoil.cu)

#Compile specifically for H100's on Kestrel
set_target_properties(airfoil_grad PROPERTIES
    CUDA_ARCHITECTURES "90"
)

# Green-Gauss gradient on airfoil - linear interpolation
add_executable(airfoil_grad2 airfoil.cu)

#Compile specifically for H100's on Kestrel
set_target_properties(airfoil_grad2 PROPERTIES
    CUDA_ARCHITECTURES "90"
)

