cmake_minimum_required(VERSION 3.31)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -ffast-math -march=skylake-avx512")

project(Collection)

add_subdirectory(src)
add_subdirectory(thirdparty)


