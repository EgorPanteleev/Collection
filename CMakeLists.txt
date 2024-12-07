cmake_minimum_required(VERSION 3.29)
set(CMAKE_CXX_STANDARD 17)

project(Collection)

find_package(Kokkos REQUIRED)

enable_testing()

add_subdirectory(BVH)

#add_subdirectory(Models)

add_subdirectory(Externals)

add_subdirectory(ThreadsSheduler)

add_subdirectory(OBJReader)

add_subdirectory(Math)

add_subdirectory(Containers)

add_subdirectory(Mesh)

add_subdirectory(Refactor)

add_subdirectory(Refactor/Application)

add_subdirectory(Utils)

#add_subdirectory(Shapes)

#add_subdirectory(Render/General)

#add_subdirectory(Render/Rasterizer)

#add_subdirectory(Render/RayTracer)

#add_subdirectory(Applications)

add_subdirectory(Tests)

