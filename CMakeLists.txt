cmake_minimum_required(VERSION 3.15)
project(P-CGPDE)

set(CMAKE_CXX_STANDARD 17)
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -static-libstdc++ -static-libgcc")

#include_directories(.)
#include_directories(include)
#link_directories(lib/x64)

#-lm2s-opencl -I/home/bmarcosps/multi2sim-5.0/runtime/include
include_directories(.)
include_directories("C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.0\\include")
link_directories("C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.0\\lib\\Win32")

#include_directories("/mnt/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.0/include")
#link_directories("/mnt/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.0/lib/Win32")
#link_directories("/usr/lib/x86_64-linux-gnu/")

#include_directories("/home/bmarcosps/multi2sim-5.0/runtime/include")
#link_directories("/home/bmarcosps/multi2sim-5.0/lib/.libs")

add_executable(P-CGPDE main.cpp cgp.cpp cgp.h constants.h utils.cpp utils.h stack.cpp stack.h GPTime.cpp GPTime.h OCLConfig.cpp OCLConfig.h)
target_link_libraries(P-CGPDE OpenCL)