
# ADI in CUDA
add_executable(adi_cuda try_adi.cu)

#Compile specifically for H100's on Kestrel
set_target_properties(adi_cuda PROPERTIES
    CUDA_ARCHITECTURES "90"
)


