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

add_compile_definitions(PROJECT_PATH="${CMAKE_SOURCE_DIR}/")
add_compile_definitions(BINARY_PATH="${CMAKE_BINARY_DIR}/")
add_compile_definitions(ASSETS_PATH="${CMAKE_SOURCE_DIR}/assets/")
add_compile_definitions(SCENES_PATH="${CMAKE_SOURCE_DIR}/assets/scenes/")
add_compile_definitions(MODELS_PATH="${CMAKE_SOURCE_DIR}/assets/models/")
#add_compile_definitions(COMPILED_SHADERS_PATH="${CMAKE_BINARY_DIR}/compiled_shaders/")

add_custom_target(all_tests)
add_subdirectory(Message)
add_subdirectory(Utils)
add_subdirectory(ThreadPool)
add_subdirectory(BVH)
add_subdirectory(Camera)
add_subdirectory(ModelLoader)
add_subdirectory(Graphics)
add_subdirectory(Applications)


