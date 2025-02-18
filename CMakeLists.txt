cmake_minimum_required(VERSION 3.10)

project(stereo_reconstruction)

# Set C++ flags
set(CMAKE_CXX_STANDARD 14)

find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(FLANN REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS} ${FLANN_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIR})

# Fetch libigl
include(FetchContent)
FetchContent_Declare(
        libigl
        GIT_REPOSITORY https://github.com/libigl/libigl.git
        GIT_TAG main
)
FetchContent_MakeAvailable(libigl)

# Common headers
set(HEADERS
    includes/Eigen.h
)

# 3) Executable: stereo_reconstruction
set(SRC_STEREO
    stereo_reconstruction.cpp
    includes/fast_sgbm/fast_sgbm.cpp
    ${HEADERS}
)
add_executable(stereo_reconstruction ${SRC_STEREO})
target_link_libraries(stereo_reconstruction
    Eigen3::Eigen
    ${OpenCV_LIBS}
    flann
    lz4
    igl::core
)

