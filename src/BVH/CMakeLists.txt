project(BVH)

find_package(glm REQUIRED)

#static Library
file(GLOB SOURCES "./private/*.cpp")
file(GLOB HEADERS "./public/*.hpp")
add_library(${PROJECT_NAME} STATIC ${SOURCES} ${HEADERS})
target_include_directories(${PROJECT_NAME} PUBLIC ./public/)

set(Libs Message Utils)
set(ThirdPartyLibs)
target_link_libraries(${PROJECT_NAME} ${Libs} ${ThirdPartyLibs})

file(GLOB TESTS "./test/*.cpp")
foreach(TEST_FILE ${TESTS})
    get_filename_component(TEST_NAME ${TEST_FILE} NAME_WE)
    add_executable(${TEST_NAME} ${TEST_FILE})
    target_link_libraries(${TEST_NAME} PRIVATE gtest_main PathTracer Message ${PROJECT_NAME})
    gtest_discover_tests(${TEST_NAME})
    add_dependencies(all_tests ${TEST_NAME})
endforeach ()
