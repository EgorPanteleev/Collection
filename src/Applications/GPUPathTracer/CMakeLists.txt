find_package(Vulkan REQUIRED)
set(Libs glfw Vulkan::Vulkan Vulkan::PathTracer )
add_executable(GPUPathTracer main.cpp)
target_link_libraries(GPUPathTracer ${Libs})