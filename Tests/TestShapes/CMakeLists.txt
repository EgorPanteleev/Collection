project(TestMath)

set(Libs Shapes Math)

add_executable( TestTriangle TestTriangle.cpp )

target_link_libraries(TestTriangle ${Libs})

add_test( TestTriangle TestTriangle )


