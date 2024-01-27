cmake_minimum_required(VERSION 3.26)
set(CMAKE_CXX_STANDARD 23)

project(Collection)

enable_testing()

add_subdirectory(Math)

add_subdirectory(BitMapImage)

add_subdirectory(Shapes)

add_subdirectory(RayTracer)

add_subdirectory(Applications)

add_subdirectory(Tests)

