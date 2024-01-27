project(TestMath)

set(Libs Math)

add_executable( TestVectors TestVectors.cpp )

add_executable( TestMatrix TestMatrix.cpp )


target_link_libraries(TestVectors ${Libs})

target_link_libraries(TestMatrix ${Libs})

add_test( TestVectors TestVectors )

add_test( TestMatrix TestMatrix )

