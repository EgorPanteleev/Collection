project(RenderGeneral)

#static Library
file(GLOB SOURCES "*.cpp")
file(GLOB HEADERS "*.h")
add_library(${PROJECT_NAME} STATIC ${SOURCES} ${HEADERS})
target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

set(Libs stbImage Mesh Math )
target_link_libraries(${PROJECT_NAME} ${Libs})
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -static-libstdc++ -static-libgcc")
