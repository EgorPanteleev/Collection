project(VulkanCore)

find_package(Vulkan REQUIRED)

#static Library
file(GLOB SOURCES "./Wrap/private/*.cpp" "./private/*.cpp")
file(GLOB HEADERS "./Wrap/public/*.hpp" "./public/*.hpp" )
add_library(${PROJECT_NAME} STATIC ${SOURCES} ${HEADERS})
target_include_directories(${PROJECT_NAME} PUBLIC ./Wrap/public/ ./public/)

set(Libs Message BaseWindow ModelLoader)
set(ThirdPartyLibs vma imgui)
target_link_libraries(${PROJECT_NAME} ${Libs} ${ThirdPartyLibs})
add_library(Vulkan::Core ALIAS VulkanCore)