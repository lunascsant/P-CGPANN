cmake_minimum_required(VERSION 3.15)
project(P-CGPDE)

set(CMAKE_CXX_STANDARD 17)
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -static-libstdc++ -static-libgcc")

find_package(CUDA)
if(CUDA_FOUND)
    set(OPENCL_INC_DIR ${CUDA_INCLUDE_DIRS})
    set(OPENCL_LIB     ${CUDA_OpenCL_LIBRARY})
    include_directories(${OPENCL_INC_DIRS})
    set(OPENCL_LIB ${CUDA_OpenCL_LIBRARY})
    set(HAVE_OPENCL 1)
    message("OpenCL_Include: ${CUDA_INCLUDE_DIRS}")
else()
    find_package(OpenCL REQUIRED)
    if(OPENCL_FOUND)
        message("OpenCL found")
        message("OpenCL_VERSION_MAJOR: ${OpenCL_VERSION_MAJOR}")
        message("OpenCL_VERSION_MINOR: ${OpenCL_VERSION_MINOR}")
        set(OPENCL_INC_DIR ${OpenCL_INCLUDE_DIRS})
        include_directories(${OpenCL_INCLUDE_DIRS})
        set(OPENCL_LIB ${OpenCL_LIBRARIES})
        set(HAVE_OPENCL 1)
    endif()
endif()

include_directories(${OPENCL_INC_DIR})

file(GLOB SRC_FILES *.cpp *.h)

add_executable(pcgp ${SRC_FILES})
target_link_libraries(pcgp PUBLIC ${OPENCL_LIB})

