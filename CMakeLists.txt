cmake_minimum_required(VERSION 3.15)
project(P-CGPDE)

set(CMAKE_CXX_STANDARD 17)
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -static-libstdc++ -static-libgcc -fopenmp")

#include_directories(.)
#include_directories(include)
#link_directories(lib/x64)


include_directories(.)
include_directories("C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.0\\include")
link_directories("C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.0\\lib\\x64")

add_executable(P-CGPDE main.cpp cgp.cpp cgp.h constants.h utils.cpp utils.h stack.cpp stack.h GPTime.cpp GPTime.h OCLConfig.cpp OCLConfig.h)
target_link_libraries(P-CGPDE OpenCL)