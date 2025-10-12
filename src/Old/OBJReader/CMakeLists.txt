project(OBJReader)

#static Library
file(GLOB SOURCES "*.cpp")
file(GLOB HEADERS "*.h")
add_library(${PROJECT_NAME} STATIC ${SOURCES} ${HEADERS})
target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

set(Libs Refactor Math)
target_link_libraries(${PROJECT_NAME} ${Libs})

if ( CMAKE_CXX_COMPILER MATCHES ".*hipcc$" )
    message( STATUS "HIP is enabled" )
    find_package( hip REQUIRED )
    target_compile_definitions( ${PROJECT_NAME} PRIVATE HIP_ENABLED )
else()
    message( STATUS "HIP is not enabled" )
endif()