set(Libs PathTracer Utils glfw GL GLEW )

add_executable( CPUPathTracer main.cpp GLUtils.cpp PathTracerApp.cpp )

target_link_libraries(CPUPathTracer ${Libs} )