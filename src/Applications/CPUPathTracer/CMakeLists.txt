set(Libs PathTracer glfw GL GLEW )

add_executable( CPUPathTracer CPUPathTracer.cpp GLUtils.cpp )

target_link_libraries(CPUPathTracer ${Libs} )