add_custom_target(
        PathTracerCompileShaders
        COMMAND ${CMAKE_COMMAND} -E make_directory "${COMPILED_SHADERS_DIR}"
        COMMAND ${SHADER_COMPILER} "${SHADERS_DIR}" "${COMPILED_SHADERS_DIR}"

        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        COMMENT "Compiling shaders..."
)

add_dependencies(VulkanPathTracer PathTracerCompileShaders)