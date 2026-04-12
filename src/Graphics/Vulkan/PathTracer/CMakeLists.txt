project(VulkanPathTracer)

#static Library
file(GLOB SOURCES "./private/*.cpp")
file(GLOB HEADERS "./public/*.hpp")
add_library(${PROJECT_NAME} STATIC ${SOURCES} ${HEADERS})
target_include_directories(${PROJECT_NAME} PUBLIC ./public/)

set(Libs Message Vulkan::Core)
target_link_libraries(${PROJECT_NAME} ${Libs})
add_library(Vulkan::PathTracer ALIAS VulkanPathTracer)