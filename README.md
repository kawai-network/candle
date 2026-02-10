# go-candle

Golang bindings for [candle](https://github.com/huggingface/candle), bringing Hugging Face ML models to Go with zero Python dependency.

## Features

- **Text Generation**: Autoregressive text generation with SmolLM, Phi3, Llama, Mistral, Gemma, Qwen2, Falcon, and StableLM.
- **Embeddings**: Sentence embeddings with BERT and sentence-transformers (single and batch).
- **Image Classification**: Classify images with ViT and SegFormer models.
- **CLIP**: Image-text similarity scoring and image/text embeddings.
- **Depth Estimation**: Monocular depth estimation with Depth Anything V2 + DINOv2.
- **Semantic Segmentation**: Per-pixel classification with SegFormer.
- **Speech Recognition**: Transcribe audio with OpenAI Whisper (with timestamps).
- **T5 Seq2Seq**: Translation, summarization, and more with T5 and Flan-T5 models.
- **Translation**: Neural machine translation with Marian MT (fr-en, en-fr, en-es, en-zh, en-ru, en-hi).
- **Self-Contained**: Bundles the Rust binding as a compressed shared library via `go:embed`.

## Prerequisites

- **Linux** (amd64) or **macOS** (ARM64 / amd64)
- **Go** 1.24+
- **Rust** (stable toolchain) — only needed to rebuild the binding from source

## Install

```bash
go get github.com/soundprediction/go-candle@latest
```

## Usage

### Initialization

The library initializes automatically on import. Models are downloaded from [Hugging Face Hub](https://huggingface.co/) on first use and cached locally.

```go
package main

import (
    "fmt"
    "log"

    "github.com/soundprediction/go-candle/pkg/candle"
)

func main() {
    if err := candle.Init(); err != nil {
        log.Fatalf("Failed to initialize: %v", err)
    }
    fmt.Println("candle version:", candle.Version())
}
```

### Text Generation

```go
pipeline, _ := candle.NewTextGenerationPipeline(candle.TextGenerationConfig{
    ModelID:   "HuggingFaceTB/SmolLM-135M",
    MaxTokens: 50,
})
defer pipeline.Close()

text, _ := pipeline.Generate("The future of AI is")
fmt.Println(text)
```

### Embeddings

```go
pipeline, _ := candle.NewEmbeddingPipeline(candle.EmbeddingConfig{
    ModelID:   "sentence-transformers/all-MiniLM-L6-v2",
    Normalize: true,
})
defer pipeline.Close()

vec, _ := pipeline.Embed("Hello world")
fmt.Printf("Dimension: %d\n", len(vec)) // 384

vecs, _ := pipeline.EmbedBatch([]string{"Hello", "World"})
fmt.Printf("Batch size: %d\n", len(vecs)) // 2
```

### Image Classification

```go
pipeline, _ := candle.NewClassificationPipeline(candle.ClassificationConfig{
    ModelID: "google/vit-base-patch16-224",
})
defer pipeline.Close()

preds, _ := pipeline.Classify("photo.jpg", 5)
for _, p := range preds {
    fmt.Printf("%s: %.4f\n", p.Label, p.Score)
}
```

### CLIP (Image-Text Similarity)

```go
pipeline, _ := candle.NewClipPipeline(candle.ClipConfig{
    ModelID: "openai/clip-vit-base-patch32",
})
defer pipeline.Close()

scores, _ := pipeline.Score("photo.jpg", []string{"a cat", "a dog", "a car"})
fmt.Printf("cat=%.4f dog=%.4f car=%.4f\n", scores[0], scores[1], scores[2])

textEmb, _ := pipeline.EmbedText("a photo of a cat")
imgEmb, _ := pipeline.EmbedImage("photo.jpg")
```

### Whisper (Speech Recognition)

```go
pipeline, _ := candle.NewWhisperPipeline(candle.WhisperConfig{
    ModelID: "openai/whisper-tiny",
})
defer pipeline.Close()

result, _ := pipeline.Transcribe("audio.wav")
fmt.Println(result.Text)
for _, seg := range result.Segments {
    fmt.Printf("[%.1f-%.1f] %s\n", seg.Start, seg.End, seg.Text)
}
```

### T5 (Translation / Summarization)

```go
pipeline, _ := candle.NewT5Pipeline(candle.T5Config{
    ModelID: "t5-small",
})
defer pipeline.Close()

// Translation
text, _ := pipeline.Generate("translate English to French: The house is wonderful.")
fmt.Println(text) // La maison est merveilleuse.

// Summarization
summary, _ := pipeline.Generate("summarize: " + longText)
fmt.Println(summary)
```

### Translation (Marian MT)

```go
pipeline, _ := candle.NewTranslationPipeline(candle.TranslationConfig{
    LanguagePair: "fr-en", // also: en-fr, en-es, en-zh, en-ru, en-hi
})
defer pipeline.Close()

result, _ := pipeline.Translate("Bonjour le monde")
fmt.Println(result) // Hello, world
```

### Depth Estimation

```go
pipeline, _ := candle.NewDepthPipeline(candle.DepthConfig{
    ModelID: "LiheYoung/depth-anything-v2-small",
})
defer pipeline.Close()

depth, _ := pipeline.EstimateDepth("photo.jpg")
fmt.Printf("Depth map: %dx%d\n", depth.Width, depth.Height)
```

### Semantic Segmentation

```go
pipeline, _ := candle.NewSegmentationPipeline(candle.SegmentationConfig{
    ModelID: "nvidia/segformer-b0-finetuned-ade-512-512",
})
defer pipeline.Close()

seg, _ := pipeline.Segment("photo.jpg")
fmt.Printf("Segmentation: %dx%d, %d labels\n", seg.Width, seg.Height, seg.NumLabels)
```

## Examples

See the [`examples/`](examples/) directory for runnable programs:

```bash
go run ./examples/translation/
```

## Build from Source

To rebuild the Rust binding:

```bash
# Linux
make rust-linux

# macOS ARM64
make rust-darwin-arm64

# Compress and embed
make compress
```

## Running Tests

```bash
# All tests (downloads models on first run)
go test -v ./pkg/candle/...

# Skip model downloads
go test -short ./pkg/candle/...
```

## Architecture

```
go-candle/
├── candle_binding/       # Rust cdylib crate (FFI layer)
│   └── src/              # Per-pipeline Rust implementations
├── pkg/candle/           # Go package (public API)
│   ├── candle.h          # Shared C header for CGO
│   ├── loader.go         # dlopen/dlsym runtime loading
│   └── lib/              # Embedded compressed .so/.dylib (via go:embed)
├── examples/             # Runnable Go examples
└── scripts/              # Build and compression scripts
```

The binding uses `dlopen`/`dlsym` (not link-time binding) with a gzip-compressed shared library embedded via `go:embed`. JSON config strings are passed across the FFI boundary to keep the C interface simple.

## License

MIT OR Apache-2.0
