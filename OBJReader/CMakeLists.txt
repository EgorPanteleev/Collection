project(OBJReader)

#static Library
file(GLOB SOURCES "*.cpp")
file(GLOB HEADERS "*.h")
add_library(${PROJECT_NAME} STATIC ${SOURCES} ${HEADERS})
target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

set(Libs Shapes Math)

add_executable( idk idk.cpp)
target_link_libraries(${PROJECT_NAME} ${Libs})
target_link_libraries(idk ${Libs})