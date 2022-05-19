cmake_minimum_required(VERSION 3.15)
project(P-CGPDE)

set(CMAKE_CXX_STANDARD 17)
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -static-libstdc++ -static-libgcc")


find_package(OpenCL REQUIRED)
if(OPENCL_FOUND)
    message("OpenCL found")
    set(HAVE_OPENCL 1)
    message("OpenCL INCLUDE: ${OPENCL_INCLUDE_DIRS}")
    message("OpenCL LIB: ${OPENCL_LIBRARIES}")
    message("OpenCL VERSION: ${OPENCL_VERSION}")
    message("OpenCL_VERSION_MAJOR: ${OpenCL_VERSION_MAJOR}")
    message("OpenCL_VERSION_MINOR: ${OpenCL_VERSION_MINOR}")

endif()

include_directories(${OPENCL_INCLUDE_DIRS})

#configure_file(${PROJECT_SOURCE_DIR}/config.h.in ${PROJECT_SOURCE_DIR}/config.h)

file(GLOB source_files *.cpp *.h)

add_executable(P-CGPDE ${source_files})
target_link_libraries(P-CGPDE  PUBLIC OpenCL)