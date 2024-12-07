set(Libs stbImage Refactor )

add_executable( refactor main.cpp )

add_executable( HIP hip.cpp )

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

