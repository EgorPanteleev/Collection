#!/bin/bash

SHADER_DIR=$1
OUTPUT_DIR=$2

mkdir -p "$OUTPUT_DIR"

shopt -s nullglob
SHADER_FILES=("$SHADER_DIR"/*.slang)

if [ ${#SHADER_FILES[@]} -eq 0 ]; then
    echo "No .slang files found in $SHADER_DIR"
    exit 0
fi

for SHADER_FILE in "${SHADER_FILES[@]}"; do
    BASENAME=$(basename "$SHADER_FILE")
    OUTPUT_FILE="$OUTPUT_DIR/$BASENAME.spv"

    if /usr/bin/slangc "$SHADER_FILE" \
        -target spirv \
        -profile spirv_1_6 \
        -matrix-layout-column-major \
        -fvk-use-entrypoint-name \
        -O3 \
        -o "$OUTPUT_FILE"; then
        echo "Success: $SHADER_FILE compiled!"
    else
        echo "Error: Failed to compile $SHADER_FILE" >&2
        exit 1
    fi
done

echo "Slang shader compilation complete!"
