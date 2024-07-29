project(Graphics)

set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -static-libstdc++ -static-libgcc")

#set(OpenGlLibs freeglut_static opengl32 glu32 winmm )

set(Libs RayTracer Mesh Image)

add_executable( 2DView 2DView.cpp )

add_executable( 3DView 3DView.cpp )

add_executable( GUI GUI.cpp )

target_link_libraries(2DView ${Libs} Kokkos::kokkos )

target_link_libraries(3DView Kokkos::kokkos)

target_link_libraries(GUI glfw Imgui stbImage ${GLEW_LIB} Kokkos::kokkos ${Libs})

#target_link_libraries(3DView ${OpenGlLibs} ${Libs})