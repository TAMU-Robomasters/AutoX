cmake_minimum_required(VERSION 3.22)
project(autox_cuda LANGUAGES CXX CUDA)

set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)
find_package(CUDAToolkit REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Override via: cmake -DCMAKE_CUDA_ARCHITECTURES=87  (Orin is sm_87)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

pybind11_add_module(pf_cuda_cv
    src/subsystems/pf_cuda_cv.cu
)

target_link_libraries(pf_cuda_cv PRIVATE
    CUDA::cudart
    CUDA::curand
)

target_compile_options(pf_cuda_cv PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        -O3
        --expt-relaxed-constexpr
        --expt-extended-lambda
    >
)

# Install .so to site-packages root so `import pf_cuda_cv` works anywhere.
install(TARGETS pf_cuda_cv DESTINATION .)
