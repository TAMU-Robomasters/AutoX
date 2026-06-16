cmake_minimum_required(VERSION 3.22)
project(autox_cuda LANGUAGES CXX)

set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(CheckLanguage)
check_language(CUDA)
find_package(CUDAToolkit QUIET)

if(CMAKE_CUDA_COMPILER AND CUDAToolkit_FOUND)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)

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
else()
    message(STATUS "CUDA toolkit not found - skipping pf_cuda_cv extension. "
                   "Python code that imports pf_cuda_cv will fail at import time.")
endif()
