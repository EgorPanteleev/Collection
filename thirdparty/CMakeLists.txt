#STBImage
add_library(stbImage INTERFACE)
target_include_directories(stbImage INTERFACE ./stbi/)

#ImGui
file(GLOB IMGUI_SOURCES "./imgui/*.cpp" "./imgui/backends/*vulkan.cpp" "./imgui/backends/*glfw.cpp")
file(GLOB IMGUI_HEADERS "./imgui/*.h" "./imgui/backends/*vulkan.h" "./imgui/backends/*glfw.h")
add_library(imgui STATIC ${IMGUI_SOURCES} ${IMGUI_HEADERS})
target_include_directories(imgui PUBLIC ./imgui/)

#gli
add_subdirectory(./gli)

#MeshOptimizer
add_subdirectory(./meshoptimizer)

#Assimp
set(ASSIMP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(ASSIMP_BUILD_ASSIMP_TOOLS OFF CACHE BOOL "" FORCE)
set(ASSIMP_BUILD_SAMPLES OFF CACHE BOOL "" FORCE)
set(ASSIMP_INJECT_DEBUG_POSTFIX OFF CACHE BOOL "" FORCE)
set(ASSIMP_INSTALL OFF CACHE BOOL "" FORCE)
set(ASSIMP_WARNINGS_AS_ERRORS OFF CACHE BOOL "" FORCE)
add_subdirectory(./assimp)

#VulkanMemoryAllocator
add_library(vma INTERFACE)
target_include_directories(vma INTERFACE ./VulkanMemoryAllocator/include/)


