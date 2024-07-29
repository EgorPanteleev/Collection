# find_package(ALSA REQUIRED)

if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    message("Configuring for windows...")
    message("Libraries path is - ${WINDOWS_LIBRARIES}")

    set(CMAKE_PREFIX_PATH  "${WINDOWS_LIBRARIES}/glew-2.1.0")
    find_package(GLEW REQUIRED)
    set(GLEW_LIB ${GLEW_STATIC_LIBRARY_RELEASE} opengl32.lib PARENT_SCOPE)

    # Your relative paths
    set(RELATIVE_INCLUDE_DIR  "${WINDOWS_LIBRARIES}/portaudio/include")
    set(RELATIVE_LIB  "${WINDOWS_LIBRARIES}/portaudio_build/Debug/portaudio.lib")

    # Convert to absolute paths
    get_filename_component(ABSOLUTE_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/${RELATIVE_INCLUDE_DIR}" ABSOLUTE)
    get_filename_component(ABSOLUTE_LIB "${CMAKE_CURRENT_SOURCE_DIR}/${RELATIVE_LIB}" ABSOLUTE)

    # Set absolute paths with respect to the parent scope
    set(PORTAUDIO_INCLUDE_DIR "${ABSOLUTE_INCLUDE_DIR}" PARENT_SCOPE)
    set(PORTAUDIO_LIB "${ABSOLUTE_LIB}" PARENT_SCOPE)
else()
    find_package(GLEW REQUIRED)
    set(GLEW_LIB ${GLEW_LIBRARIES} GL PARENT_SCOPE)

    set(PORTAUDIO_LIB portaudio PARENT_SCOPE)
endif()

#add_subdirectory(glew/build/cmake/)
#target_compile_definitions(glew_s PUBLIC GLEW_NO_GLU)

#add_subdirectory(unittest-cpp)

add_subdirectory(glfw)

project(Imgui)
set(${PROJECT_NAME}_SOURCES
        imgui/imgui.cpp
        imgui/imgui_draw.cpp
        imgui/imgui_tables.cpp
        imgui/imgui_widgets.cpp

        imgui/backends/imgui_impl_glfw.cpp
        imgui/backends/imgui_impl_opengl3.cpp
)

add_library(${PROJECT_NAME} STATIC ${${PROJECT_NAME}_SOURCES})
include_directories(${PROJECT_NAME} ./glfw/include)
target_include_directories(${PROJECT_NAME} PUBLIC ./imgui/ ./imgui/backends/)

project(stbImage)
add_library(${PROJECT_NAME} INTERFACE)
target_include_directories(${PROJECT_NAME} INTERFACE ./stb/)

