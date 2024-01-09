project(Graphics)

set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -static-libstdc++ -static-libgcc")

set(OpenGlLibs freeglut_static opengl32 glu32 winmm )

set(Libs Shapes)

add_executable( View View.cpp )

target_link_libraries(View ${OpenGlLibs} ${Libs})