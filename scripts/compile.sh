#!/bin/bash

# Directory containing shader files
SHADER_DIR=$1

# Output directory for compiled shaders (optional)
OUTPUT_DIR=$2

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Find all .vert and .frag files in the shader directory
for SHADER_FILE in "$SHADER_DIR"/*.vert "$SHADER_DIR"/*.frag "$SHADER_DIR"/*.comp "$SHADER_DIR"/*.rgen "$SHADER_DIR"/*.rmiss "$SHADER_DIR"/*.rchit; do
    if [ -f "$SHADER_FILE" ]; then
        # Get the base name of the shader file (e.g., SimpleShader.frag)
        BASENAME=$(basename "$SHADER_FILE")

        # Define the output file name (e.g., SimpleShader.frag.spv)
        OUTPUT_FILE="$OUTPUT_DIR/$BASENAME.spv"

        # Compile the shader
        #echo "Compiling $SHADER_FILE to $OUTPUT_FILE..."
        if /usr/bin/glslc --target-env=vulkan1.3 -O "$SHADER_FILE" -o "$OUTPUT_FILE"; then
          echo "Success: $SHADER_FILE compiled!"
        else
           echo "Error: Failed to compile $SHADER_FILE" >&2
           exit 1
        fi
    fi
done

echo "Shader compilation complete!"