project(Refactor)

#static Library
file(GLOB SOURCES "*.cpp")
file(GLOB HEADERS "*.h")
add_library(${PROJECT_NAME} STATIC ${SOURCES} ${HEADERS})
target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

find_package(Lua REQUIRED)

#TODO add if lua found etc

set(Libs Math Containers ${LUA_LIBRARIES} )
target_link_libraries(${PROJECT_NAME} ${Libs} )
target_compile_options( ${PROJECT_NAME} PRIVATE -fgpu-rdc )


if ( CMAKE_CXX_COMPILER MATCHES ".*hipcc$" )
    message( STATUS "HIP is enabled" )
    find_package( hip REQUIRED )
    target_link_libraries( ${PROJECT_NAME} hip::host hip::device )
    target_compile_definitions( ${PROJECT_NAME} PRIVATE HIP_ENABLED )
else()
    message( STATUS "HIP is not enabled" )
endif()
