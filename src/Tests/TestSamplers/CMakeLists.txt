project(TestSamplers)

set(Libs RayTracer Math gtest gtest_main )

add_executable( TestSamplers TestSamplers.cpp )

target_link_libraries( TestSamplers ${Libs})




