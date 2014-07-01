cmake_minimum_required (VERSION 2.8)

set(PROJECT_NAME toy_problem)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_LIST_DIR}/cmake")
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)

add_definitions(-std=c++11)

find_package(OpenCL REQUIRED)

include_directories(${OpenCL_INCLUDE_DIRS})

add_executable(${PROJECT_NAME} main.cpp clsvm.hpp clsvm.cpp)

target_link_libraries(${PROJECT_NAME} matio ${OPENCL_LIBRARIES}) 
