#!/bin/bash

# Build the candle_binding shared library for a target platform.
# Usage: ./scripts/build-binding.sh <platform>
# Platforms: linux-amd64, darwin-arm64, darwin-amd64

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

PLATFORM="${1:-}"
if [ -z "$PLATFORM" ]; then
    echo "Usage: $0 <platform>"
    echo "Platforms: linux-amd64, darwin-arm64, darwin-amd64"
    exit 1
fi

cd "$PROJECT_ROOT/candle_binding"

case "$PLATFORM" in
    linux-amd64)
        TARGET_TRIPLE="x86_64-unknown-linux-gnu"
        LIB_NAME="libcandle_binding.so"
        TARGET_DIR="$PROJECT_ROOT/pkg/candle/lib/linux-amd64"
        echo "Building for Linux amd64..."
        cargo build --release --target "$TARGET_TRIPLE"
        SOURCE_LIB="target/$TARGET_TRIPLE/release/$LIB_NAME"
        ;;
    darwin-arm64)
        TARGET_TRIPLE="aarch64-apple-darwin"
        LIB_NAME="libcandle_binding.dylib"
        TARGET_DIR="$PROJECT_ROOT/pkg/candle/lib/darwin-arm64"
        echo "Building for macOS ARM64..."
        if command -v rustup >/dev/null 2>&1; then
            rustup target add "$TARGET_TRIPLE" 2>/dev/null || true
        fi
        cargo build --release --target "$TARGET_TRIPLE"
        SOURCE_LIB="target/$TARGET_TRIPLE/release/$LIB_NAME"
        ;;
    darwin-amd64)
        TARGET_TRIPLE="x86_64-apple-darwin"
        LIB_NAME="libcandle_binding.dylib"
        TARGET_DIR="$PROJECT_ROOT/pkg/candle/lib/darwin-amd64"
        echo "Building for macOS amd64..."
        if command -v rustup >/dev/null 2>&1; then
            rustup target add "$TARGET_TRIPLE" 2>/dev/null || true
        fi
        cargo build --release --target "$TARGET_TRIPLE"
        SOURCE_LIB="target/$TARGET_TRIPLE/release/$LIB_NAME"
        ;;
    host)
        # Build for the current host, no cross-compilation
        LIB_NAME=""
        case "$(uname -s)-$(uname -m)" in
            Linux-x86_64)
                LIB_NAME="libcandle_binding.so"
                TARGET_DIR="$PROJECT_ROOT/pkg/candle/lib/linux-amd64"
                ;;
            Darwin-arm64)
                LIB_NAME="libcandle_binding.dylib"
                TARGET_DIR="$PROJECT_ROOT/pkg/candle/lib/darwin-arm64"
                ;;
            Darwin-x86_64)
                LIB_NAME="libcandle_binding.dylib"
                TARGET_DIR="$PROJECT_ROOT/pkg/candle/lib/darwin-amd64"
                ;;
            *)
                echo "Unsupported host platform: $(uname -s)-$(uname -m)"
                exit 1
                ;;
        esac
        echo "Building for host platform..."
        cargo build --release
        SOURCE_LIB="target/release/$LIB_NAME"
        ;;
    *)
        echo "Unknown platform: $PLATFORM"
        echo "Platforms: linux-amd64, darwin-arm64, darwin-amd64, host"
        exit 1
        ;;
esac

if [ ! -f "$SOURCE_LIB" ]; then
    echo "Build failed: $SOURCE_LIB not found"
    exit 1
fi

mkdir -p "$TARGET_DIR"

# Compress with gzip
echo "Compressing $LIB_NAME..."
gzip -9 -c "$SOURCE_LIB" > "$TARGET_DIR/${LIB_NAME}.gz"

echo "Built: $TARGET_DIR/${LIB_NAME}.gz"
echo "Size: $(du -h "$TARGET_DIR/${LIB_NAME}.gz" | cut -f1)"
