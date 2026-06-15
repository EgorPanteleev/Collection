#!/bin/bash
set -e
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "$2"

${SCRIPT_DIR}/compile_slang.sh $1 $2 | tee "$2/slang_compile.log"
