project(VulkanPathTracer)

find_package(Vulkan REQUIRED)

set(SHADERS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/shaders")
set(COMPILED_SHADERS_DIR "${CMAKE_CURRENT_BINARY_DIR}/compiled_shaders")
add_compile_definitions(SHADERS_DIR="${SHADERS_DIR}")
add_compile_definitions(COMPILED_SHADERS_DIR="${COMPILED_SHADERS_DIR}")

#static Library
file(GLOB SOURCES "./private/*.cpp")
file(GLOB HEADERS "./public/*.hpp")
add_library(${PROJECT_NAME} STATIC ${SOURCES} ${HEADERS})
target_include_directories(${PROJECT_NAME} PUBLIC ./public/ ${CMAKE_SOURCE_DIR}/assets/fonts/)

set(Libs Message Utils Geometry Camera BVH Vulkan::Core nlohmann_json::nlohmann_json)
target_link_libraries(${PROJECT_NAME} ${Libs})
add_library(Vulkan::PathTracer ALIAS VulkanPathTracer)
add_subdirectory(shaders)

#applications
set(AppLibs Message Geometry ModelLoader Camera glfw Vulkan::Vulkan Vulkan::PathTracer )
add_executable(PathTracerGPU applications/PathTracer.cpp)
target_link_libraries(PathTracerGPU ${AppLibs})