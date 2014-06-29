cmake_minimum_required (VERSION 2.8)

set(PROJECT_NAME toy_problem)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_LIST_DIR}/cmake")

find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(OpenCL REQUIRED)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)

add_definitions(-std=c++11)

include_directories(${OpenCV_INCLUDE_DIRS} ${OpenCL_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIR})

add_executable(${PROJECT_NAME} main.cpp clsvm.hpp)

target_link_libraries(${PROJECT_NAME} matio ${OpenCV_LIBS} ${OPENCL_LIBRARIES}) 
