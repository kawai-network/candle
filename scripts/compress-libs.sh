#!/bin/bash

# Re-compress existing binaries in pkg/candle/lib/
# Useful after manually updating binaries.

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

LIB_DIR="$PROJECT_ROOT/pkg/candle/lib"

for platform_dir in "$LIB_DIR"/*/; do
    platform=$(basename "$platform_dir")
    echo "Processing $platform..."

    for lib in "$platform_dir"*.so "$platform_dir"*.dylib; do
        if [ -f "$lib" ]; then
            echo "  Compressing $(basename "$lib")..."
            gzip -9 -f "$lib"
        fi
    done
done

echo "Done."
