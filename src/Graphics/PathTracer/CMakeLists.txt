project(PathTracer)

#static Library
file(GLOB SOURCES "./private/*.cpp")
file(GLOB HEADERS "./public/*.hpp")
add_library(${PROJECT_NAME} STATIC ${SOURCES} ${HEADERS})
target_include_directories(${PROJECT_NAME} PUBLIC ./public/)

set(Libs Camera Message Utils)
set(ThirdPartyLibs)
target_link_libraries(${PROJECT_NAME} ${Libs} ${ThirdPartyLibs})