cmake_minimum_required(VERSION 3.31)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -ffast-math -march=skylake-avx512")
set(SCRIPTS_DIR ${CMAKE_SOURCE_DIR}/scripts)
set(SHADER_COMPILER "${SCRIPTS_DIR}/compile_helper.sh")
set(SLANG_SHADER_COMPILER "${SCRIPTS_DIR}/compile_slang_helper.sh")

project(Collection)

add_subdirectory(src)
add_subdirectory(thirdparty)


