cmake_minimum_required(VERSION 3.26)
set(CMAKE_CXX_STANDARD 23)

project(Collection)

enable_testing()

add_subdirectory(BVH)

#add_subdirectory(Models)

add_subdirectory(ThreadsSheduler)

add_subdirectory(OBJReader)

add_subdirectory(Math)

add_subdirectory(BitMapImage)

add_subdirectory(Mesh)

#add_subdirectory(Shapes)

add_subdirectory(RayTracer)

add_subdirectory(Applications)

add_subdirectory(Tests)

