cmake_minimum_required(VERSION 3.18)
project(bm3d_gpu LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Silence warnings about unknown compiler features with older CMake
if(POLICY CMP0125)
  cmake_policy(SET CMP0125 NEW)
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

set(SRC
    src/blockmatching.cu
    src/dct8x8.cu
    src/filtering.cu
    Bm3dGpu.cpp
)

add_library(bm3d_gpu STATIC ${SRC})

target_link_libraries(bm3d_gpu PUBLIC cudart cufft)

target_include_directories(bm3d_gpu PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>)

find_package(OpenCV REQUIRED)

target_link_libraries(bm3d_gpu PUBLIC ${OpenCV_LIBS})
target_include_directories(bm3d_gpu PUBLIC ${OpenCV_INCLUDE_DIRS})

add_executable(demo demo.cpp)

target_link_libraries(demo PRIVATE bm3d_gpu ${OpenCV_LIBS})
