package candle

/*
#cgo LDFLAGS: -ldl
#include <stdlib.h>
#include <dlfcn.h>
#include <stdio.h>
#include "candle.h"

void* open_lib(const char* path) {
    return dlopen(path, RTLD_LAZY | RTLD_GLOBAL);
}

void* get_sym(void* handle, const char* name) {
    return dlsym(handle, name);
}

char* get_dlerror() {
    return dlerror();
}
*/
import "C"
import (
	"compress/gzip"
	"embed"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"unsafe"
)

//go:embed lib
var libFS embed.FS

var (
	initialized bool
	dlHandle    unsafe.Pointer
)

// Init loads the candle binding shared library. It is safe to call multiple times;
// subsequent calls are no-ops. Called automatically via init().
//
// If CANDLE_LIB_PATH environment variable is set, it will load the library from
// that path instead of the embedded one. This is useful for testing with custom builds.
func Init() error {
	if initialized {
		return nil
	}

	// Check for custom library path (for testing with custom builds)
	if customPath := os.Getenv("CANDLE_LIB_PATH"); customPath != "" {
		return initFromPath(customPath)
	}

	goOS := runtime.GOOS
	goArch := runtime.GOARCH

	var libPath string
	var libName string

	switch goOS {
	case "darwin":
		switch goArch {
		case "arm64":
			libPath = "lib/darwin-arm64"
		case "amd64":
			libPath = "lib/darwin-amd64"
		default:
			return fmt.Errorf("unsupported darwin architecture: %s", goArch)
		}
		libName = "libcandle_binding.dylib.gz"
	case "linux":
		if goArch == "amd64" {
			libPath = "lib/linux-amd64"
		} else {
			return fmt.Errorf("unsupported platform: %s/%s", goOS, goArch)
		}
		libName = "libcandle_binding.so.gz"
	default:
		return fmt.Errorf("unsupported platform: %s/%s", goOS, goArch)
	}

	tmpDir, err := os.MkdirTemp("", "go-candle-lib")
	if err != nil {
		return fmt.Errorf("failed to create temp dir: %w", err)
	}

	// Extract and decompress the binding library
	destName := strings.TrimSuffix(libName, ".gz")
	destPath := filepath.Join(tmpDir, destName)
	if err := extractAndDecompress(filepath.Join(libPath, libName), destPath); err != nil {
		return fmt.Errorf("failed to extract binding library: %w", err)
	}

	// dlopen the binding library
	cPath := C.CString(destPath)
	defer C.free(unsafe.Pointer(cPath))
	dlHandle = C.open_lib(cPath)
	if dlHandle == nil {
		cErr := C.get_dlerror()
		return fmt.Errorf("dlopen failed: %s", C.GoString(cErr))
	}

	// Load all symbols
	if err := loadAllSymbols(); err != nil {
		return err
	}

	initialized = true
	return nil
}

// initFromPath loads the library from a custom path (for testing with custom builds).
func initFromPath(libPath string) error {
	// Check if file exists
	if _, err := os.Stat(libPath); os.IsNotExist(err) {
		return fmt.Errorf("library not found: %s", libPath)
	}

	// dlopen the library
	cPath := C.CString(libPath)
	defer C.free(unsafe.Pointer(cPath))
	dlHandle = C.open_lib(cPath)
	if dlHandle == nil {
		cErr := C.get_dlerror()
		return fmt.Errorf("dlopen failed: %s", C.GoString(cErr))
	}

	// Load all symbols
	if err := loadAllSymbols(); err != nil {
		return err
	}

	initialized = true
	return nil
}

func loadSym(name string) (unsafe.Pointer, error) {
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	sym := C.get_sym(dlHandle, cName)
	if sym == nil {
		return nil, fmt.Errorf("symbol not found: %s", name)
	}
	return sym, nil
}

func loadAllSymbols() error {
	var err error

	// candle_last_error
	if fnCandleLastError, err = loadSym("candle_last_error"); err != nil {
		return err
	}
	// candle_binding_version
	if fnCandleBindingVersion, err = loadSym("candle_binding_version"); err != nil {
		return err
	}

	// Text Generation
	if fnNewTextGenerationPipeline, err = loadSym("new_text_generation_pipeline"); err != nil {
		return err
	}
	if fnRunTextGeneration, err = loadSym("run_text_generation"); err != nil {
		return err
	}
	if fnFreeTextGenerationPipeline, err = loadSym("free_text_generation_pipeline"); err != nil {
		return err
	}
	if fnFreeTextGenerationResult, err = loadSym("free_text_generation_result"); err != nil {
		return err
	}

	// Embeddings
	if fnNewEmbeddingPipeline, err = loadSym("new_embedding_pipeline"); err != nil {
		return err
	}
	if fnRunEmbedding, err = loadSym("run_embedding"); err != nil {
		return err
	}
	if fnRunEmbeddingBatch, err = loadSym("run_embedding_batch"); err != nil {
		return err
	}
	if fnFreeEmbeddingPipeline, err = loadSym("free_embedding_pipeline"); err != nil {
		return err
	}
	if fnFreeEmbeddingResult, err = loadSym("free_embedding_result"); err != nil {
		return err
	}
	if fnFreeBatchEmbeddingResult, err = loadSym("free_batch_embedding_result"); err != nil {
		return err
	}

	// Classification
	if fnNewClassificationPipeline, err = loadSym("new_classification_pipeline"); err != nil {
		return err
	}
	if fnRunClassification, err = loadSym("run_classification"); err != nil {
		return err
	}
	if fnFreeClassificationPipeline, err = loadSym("free_classification_pipeline"); err != nil {
		return err
	}
	if fnFreeClassificationResult, err = loadSym("free_classification_result"); err != nil {
		return err
	}

	// CLIP
	if fnNewClipPipeline, err = loadSym("new_clip_pipeline"); err != nil {
		return err
	}
	if fnRunClipScore, err = loadSym("run_clip_score"); err != nil {
		return err
	}
	if fnRunClipEmbedImage, err = loadSym("run_clip_embed_image"); err != nil {
		return err
	}
	if fnRunClipEmbedText, err = loadSym("run_clip_embed_text"); err != nil {
		return err
	}
	if fnFreeClipPipeline, err = loadSym("free_clip_pipeline"); err != nil {
		return err
	}
	if fnFreeClipScoreResult, err = loadSym("free_clip_score_result"); err != nil {
		return err
	}
	if fnFreeClipEmbeddingResult, err = loadSym("free_clip_embedding_result"); err != nil {
		return err
	}

	// Depth Estimation
	if fnNewDepthPipeline, err = loadSym("new_depth_pipeline"); err != nil {
		return err
	}
	if fnRunDepthEstimation, err = loadSym("run_depth_estimation"); err != nil {
		return err
	}
	if fnFreeDepthPipeline, err = loadSym("free_depth_pipeline"); err != nil {
		return err
	}
	if fnFreeDepthResult, err = loadSym("free_depth_result"); err != nil {
		return err
	}

	// Segmentation
	if fnNewSegmentationPipeline, err = loadSym("new_segmentation_pipeline"); err != nil {
		return err
	}
	if fnRunSegmentation, err = loadSym("run_segmentation"); err != nil {
		return err
	}
	if fnFreeSegmentationPipeline, err = loadSym("free_segmentation_pipeline"); err != nil {
		return err
	}
	if fnFreeSegmentationResult, err = loadSym("free_segmentation_result"); err != nil {
		return err
	}

	// Whisper
	if fnNewWhisperPipeline, err = loadSym("new_whisper_pipeline"); err != nil {
		return err
	}
	if fnRunWhisperTranscribe, err = loadSym("run_whisper_transcribe"); err != nil {
		return err
	}
	if fnFreeWhisperPipeline, err = loadSym("free_whisper_pipeline"); err != nil {
		return err
	}
	if fnFreeWhisperResult, err = loadSym("free_whisper_result"); err != nil {
		return err
	}

	// T5
	if fnNewT5Pipeline, err = loadSym("new_t5_pipeline"); err != nil {
		return err
	}
	if fnRunT5Generate, err = loadSym("run_t5_generate"); err != nil {
		return err
	}
	if fnFreeT5Pipeline, err = loadSym("free_t5_pipeline"); err != nil {
		return err
	}
	if fnFreeT5Result, err = loadSym("free_t5_result"); err != nil {
		return err
	}

	// Translation (Marian MT)
	if fnNewTranslationPipeline, err = loadSym("new_translation_pipeline"); err != nil {
		return err
	}
	if fnRunTranslation, err = loadSym("run_translation"); err != nil {
		return err
	}
	if fnFreeTranslationPipeline, err = loadSym("free_translation_pipeline"); err != nil {
		return err
	}
	if fnFreeTranslationResult, err = loadSym("free_translation_result"); err != nil {
		return err
	}

	// Video symbols are optional so older embedded binaries can still initialize.
	videoAvailable = false
	if fnNewVideoPipeline, err = loadSym("new_video_pipeline"); err == nil {
		if fnRunVideoGeneration, err = loadSym("run_video_generation"); err != nil {
			fnNewVideoPipeline = nil
			return nil
		}
		if fnFreeVideoPipeline, err = loadSym("free_video_pipeline"); err != nil {
			fnNewVideoPipeline = nil
			fnRunVideoGeneration = nil
			return nil
		}
		if fnFreeVideoResult, err = loadSym("free_video_result"); err != nil {
			fnNewVideoPipeline = nil
			fnRunVideoGeneration = nil
			fnFreeVideoPipeline = nil
			return nil
		}
		if fnSaveVideoAsGif, err = loadSym("save_video_as_gif"); err != nil {
			fnNewVideoPipeline = nil
			fnRunVideoGeneration = nil
			fnFreeVideoPipeline = nil
			fnFreeVideoResult = nil
			return nil
		}
		if fnSaveVideoFrames, err = loadSym("save_video_frames"); err != nil {
			fnNewVideoPipeline = nil
			fnRunVideoGeneration = nil
			fnFreeVideoPipeline = nil
			fnFreeVideoResult = nil
			fnSaveVideoAsGif = nil
			return nil
		}
		videoAvailable = true
	}

	return nil
}

func extractAndDecompress(srcPath, destPath string) error {
	f, err := libFS.Open(srcPath)
	if err != nil {
		return fmt.Errorf("open embedded %s: %w", srcPath, err)
	}
	defer f.Close()

	var r io.Reader = f
	if strings.HasSuffix(srcPath, ".gz") {
		gz, err := gzip.NewReader(f)
		if err != nil {
			return fmt.Errorf("gzip reader %s: %w", srcPath, err)
		}
		defer gz.Close()
		r = gz
	}

	out, err := os.OpenFile(destPath, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0755)
	if err != nil {
		return fmt.Errorf("create dest %s: %w", destPath, err)
	}
	defer out.Close()

	if _, err := io.Copy(out, r); err != nil {
		return fmt.Errorf("copy %s: %w", srcPath, err)
	}
	return nil
}

func init() {
	if err := Init(); err != nil {
		fmt.Fprintf(os.Stderr, "WARNING: go-candle failed to initialize: %v\n", err)
	}
}
