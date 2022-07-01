cmake_minimum_required(VERSION 3.15)
project(P-CGPDE)

set(CMAKE_CXX_STANDARD 17)
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -static-libstdc++ -static-libgcc")

find_package(OpenCL)
if(OpenCL_FOUND)
include_directories(${OpenCL_INCLUDE_DIRS})
link_directories(${OpenCL_LIBRARY_DIRS})
message("OpenCL found")
else()
# try to find Cuda OpenCL SDK
find_package(Cuda)
  if(Cuda_FOUND)
      include_directories(${Cuda_INCLUDE_DIRS})
  else()
        message(FATAL_ERROR "OpenCL not found")
  endif()
endif()


file(GLOB SRC_FILES *.cpp *.h)

add_executable(pcgp ${SRC_FILES})
if(OPENCL_FOUND)
  target_link_libraries(pcgp OpenCL)
endif()
if(Cuda_FOUND)
  target_link_libraries(pcgp CUDA::OpenCL)
endif()

