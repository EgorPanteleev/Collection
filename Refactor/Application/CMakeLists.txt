set(Libs stbImage Refactor )

add_executable( refactor main.cpp )

add_executable( HIP hip.cpp )

add_executable( realtime realtime.cpp )

find_package(OpenGL REQUIRED)
find_package(glfw3 3.3 REQUIRED)
find_package(GLEW REQUIRED)
target_link_libraries( realtime hip::host hip::device Refactor stbImage Containers Math OpenGL::GL glfw GLEW::GLEW )

target_compile_definitions( realtime PRIVATE HIP_ENABLED )

target_link_libraries(refactor ${Libs})

if ( CMAKE_CXX_COMPILER MATCHES ".*hipcc$" )
    message( STATUS "HIP is enabled" )
    find_package( hip REQUIRED )
    target_link_libraries( HIP hip::host hip::device Refactor stbImage Containers Math )
    target_compile_definitions( HIP PRIVATE HIP_ENABLED )
    #target_compile_options( HIP PRIVATE -fgpu-rdc )
else()
    message( STATUS "HIP is not enabled" )
endif()

