enable_testing()

# Google test
include(FetchContent)
FetchContent_Declare(
        GoogleTest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG        v1.17.0
)
FetchContent_MakeAvailable(GoogleTest)
include(GoogleTest)

add_subdirectory( Message )
add_subdirectory( Utils )
add_subdirectory( BVH )
add_subdirectory( Camera )
add_subdirectory( ModelLoader )
add_subdirectory( Graphics )
add_subdirectory( Applications )


