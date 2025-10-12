cmake_minimum_required(VERSION 3.29)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_FLAGS "-O2")

project(Collection)

enable_testing()

include_directories(${CMAKE_SOURCE_DIR})
add_subdirectory(src)
add_subdirectory(thirdparty)


