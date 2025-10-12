find_package(Kokkos REQUIRED)

enable_testing()

include_directories(${CMAKE_SOURCE_DIR})
add_subdirectory(BVH)
add_subdirectory(ThreadsSheduler)
add_subdirectory(OBJReader)
add_subdirectory(Math)
add_subdirectory(Containers)
add_subdirectory(Mesh)
add_subdirectory(Memory)
add_subdirectory(Allocators)
add_subdirectory(Refactor)
add_subdirectory(Refactor/Application)
add_subdirectory(Utils)
add_subdirectory(Tests)

