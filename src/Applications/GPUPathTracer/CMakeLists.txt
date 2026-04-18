find_package(Vulkan REQUIRED)
set(Libs Geometry ModelLoader Camera glfw Vulkan::Vulkan Vulkan::PathTracer )
add_executable(GPUPathTracer main.cpp CallBacks.cpp)
target_link_libraries(GPUPathTracer ${Libs})