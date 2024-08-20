project(Graphics)

set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -static-libstdc++ -static-libgcc")

find_package(Lua REQUIRED)

set(Libs RayTracer Rasterizer Mesh)

add_executable( 2DView 2DView.cpp )

add_executable( 3DView 3DView.cpp )

add_executable( GUI GUI.cpp )

add_executable( 2DViewLua 2DViewLua.cpp)

target_link_libraries(2DView ${Libs} Kokkos::kokkos ${LUA_LIBRARIES} )

target_link_libraries(2DViewLua ${Libs} Kokkos::kokkos ${LUA_LIBRARIES} )

target_link_libraries(3DView Kokkos::kokkos)

target_link_libraries(GUI glfw Imgui ${GLEW_LIB} ${Libs} Kokkos::kokkos)

#target_link_libraries(3DView ${OpenGlLibs} ${Libs})