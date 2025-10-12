project(TestMath)

set(Libs Mesh Math)

add_executable( TestTriangle TestTriangle.cpp )

target_link_libraries(TestTriangle ${Libs})

add_test( TestTriangle TestTriangle )


