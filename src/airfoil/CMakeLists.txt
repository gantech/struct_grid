# Green-Gauss gradient on airfoil
add_executable(airfoil_grad airfoil.cu)

#Compile specifically for H100's on Kestrel
set_target_properties(airfoil_grad PROPERTIES
    CUDA_ARCHITECTURES "90"
)

# Green-Gauss gradient on airfoil - linear interpolation
add_executable(airfoil_grad2 airfoil_2.cu)

#Compile specifically for H100's on Kestrel
set_target_properties(airfoil_grad2 PROPERTIES
    CUDA_ARCHITECTURES "90"
)

# Green-Gauss gradient on airfoil - gradient extrapolation
add_executable(airfoil_grad3 airfoil_3.cu)

#Compile specifically for H100's on Kestrel
set_target_properties(airfoil_grad3 PROPERTIES
    CUDA_ARCHITECTURES "90"
)
