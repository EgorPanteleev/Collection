project(TestContainers)

set(Libs Containers gtest gtest_main )

add_executable( TestVector TestVector.cpp )

target_link_libraries( TestVector ${Libs})




