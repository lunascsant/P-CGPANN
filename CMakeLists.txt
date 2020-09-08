cmake_minimum_required(VERSION 3.15)
project(P-CGPDE)

set(CMAKE_CXX_STANDARD 17)
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -static-libstdc++ -static-libgcc")

#include_directories(.)
#include_directories(include)
#link_directories(lib/x64)


include_directories("nvidia\\include")
link_directories("nvidia\\lib\\Win32")

#include_directories("amd\\include")
#link_directories("amd\\lib\\x86")
#include_directories("C:\\Program Files (x86)\\OCL_SDK_Light\\include")
#link_directories("C:\\Program Files (x86)\\OCL_SDK_Light\\lib\\x86")


add_executable(P-CGPDE main.cpp cgp.cpp cgp.h constants.h utils.cpp utils.h stack.cpp stack.h GPTime.cpp GPTime.h OCLConfig.cpp OCLConfig.h)
target_link_libraries(P-CGPDE OpenCL)