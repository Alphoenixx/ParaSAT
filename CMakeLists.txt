cmake_minimum_required(VERSION 3.15)
project(ParaSAT_GPU LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenMP REQUIRED)

# 1. Executable
add_executable(parasat_gpu src/main.cpp)
target_include_directories(parasat_gpu PRIVATE src)
target_link_libraries(parasat_gpu PRIVATE OpenMP::OpenMP_CXX)

# 2. Shared Library (C API)
add_library(parasat_shared SHARED src/parasat_api.cpp)
target_include_directories(parasat_shared PRIVATE src include)
target_link_libraries(parasat_shared PRIVATE OpenMP::OpenMP_CXX)
set_target_properties(parasat_shared PROPERTIES OUTPUT_NAME "parasat" PREFIX "")

if(MSVC)
    target_compile_options(parasat_gpu PRIVATE /O2 /arch:AVX2)
    target_compile_options(parasat_shared PRIVATE /O2 /arch:AVX2)
else()
    set(GCC_OPTS -O3 -march=native -funroll-loops -fomit-frame-pointer)
    target_compile_options(parasat_gpu PRIVATE ${GCC_OPTS})
    target_compile_options(parasat_shared PRIVATE ${GCC_OPTS})
    
    include(CheckIPOSupported)
    check_ipo_supported(RESULT lto_supported OUTPUT error)
    if(lto_supported)
        set_property(TARGET parasat_gpu PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
        set_property(TARGET parasat_shared PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
    endif()
endif()

set_target_properties(parasat_gpu PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin")
set_target_properties(parasat_shared PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin" RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin")
