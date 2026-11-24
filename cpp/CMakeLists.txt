cmake_minimum_required(VERSION 3.16)

project(FireSceneEnhancer)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(fire_enhancer fire_enhancer.cpp)

target_include_directories(
    fire_enhancer
    PRIVATE
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(
    fire_enhancer
    PRIVATE
    ${OpenCV_LIBS}
    OpenMP::OpenMP_CXX
)