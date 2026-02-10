.PHONY: all build test clean rust-linux rust-darwin-arm64 rust-darwin-amd64 compress

# Default target
all: build

# Build Go code
build:
	go build ./...

# Run tests
test:
	go test ./...

# Run tests (skip model downloads)
test-short:
	go test -short ./...

# Compile Rust binding for Linux amd64
rust-linux:
	./scripts/build-binding.sh linux-amd64

# Compile Rust binding for macOS ARM
rust-darwin-arm64:
	./scripts/build-binding.sh darwin-arm64

# Compile Rust binding for macOS Intel
rust-darwin-amd64:
	./scripts/build-binding.sh darwin-amd64

# Compress existing binaries
compress:
	./scripts/compress-libs.sh

# Clean build artifacts
clean:
	go clean
	rm -rf candle_binding/target
