package candle

/*
#include "candle.h"
*/
import "C"
import "unsafe"

// Function pointer variables populated by loader.go Init()
var (
	// Core
	fnCandleLastError      unsafe.Pointer
	fnCandleBindingVersion unsafe.Pointer

	// Text Generation
	fnNewTextGenerationPipeline  unsafe.Pointer
	fnRunTextGeneration          unsafe.Pointer
	fnFreeTextGenerationPipeline unsafe.Pointer
	fnFreeTextGenerationResult   unsafe.Pointer

	// Embeddings
	fnNewEmbeddingPipeline     unsafe.Pointer
	fnRunEmbedding             unsafe.Pointer
	fnRunEmbeddingBatch        unsafe.Pointer
	fnFreeEmbeddingPipeline    unsafe.Pointer
	fnFreeEmbeddingResult      unsafe.Pointer
	fnFreeBatchEmbeddingResult unsafe.Pointer

	// Classification
	fnNewClassificationPipeline  unsafe.Pointer
	fnRunClassification          unsafe.Pointer
	fnFreeClassificationPipeline unsafe.Pointer
	fnFreeClassificationResult   unsafe.Pointer

	// CLIP
	fnNewClipPipeline         unsafe.Pointer
	fnRunClipScore            unsafe.Pointer
	fnRunClipEmbedImage       unsafe.Pointer
	fnRunClipEmbedText        unsafe.Pointer
	fnFreeClipPipeline        unsafe.Pointer
	fnFreeClipScoreResult     unsafe.Pointer
	fnFreeClipEmbeddingResult unsafe.Pointer

	// Depth Estimation
	fnNewDepthPipeline   unsafe.Pointer
	fnRunDepthEstimation unsafe.Pointer
	fnFreeDepthPipeline  unsafe.Pointer
	fnFreeDepthResult    unsafe.Pointer

	// Segmentation
	fnNewSegmentationPipeline  unsafe.Pointer
	fnRunSegmentation          unsafe.Pointer
	fnFreeSegmentationPipeline unsafe.Pointer
	fnFreeSegmentationResult   unsafe.Pointer

	// Whisper
	fnNewWhisperPipeline   unsafe.Pointer
	fnRunWhisperTranscribe unsafe.Pointer
	fnFreeWhisperPipeline  unsafe.Pointer
	fnFreeWhisperResult    unsafe.Pointer

	// T5
	fnNewT5Pipeline  unsafe.Pointer
	fnRunT5Generate  unsafe.Pointer
	fnFreeT5Pipeline unsafe.Pointer
	fnFreeT5Result   unsafe.Pointer

	// Translation (Marian MT)
	fnNewTranslationPipeline  unsafe.Pointer
	fnRunTranslation          unsafe.Pointer
	fnFreeTranslationPipeline unsafe.Pointer
	fnFreeTranslationResult   unsafe.Pointer

	// Video Generation
	fnNewVideoPipeline   unsafe.Pointer
	fnRunVideoGeneration unsafe.Pointer
	fnFreeVideoPipeline  unsafe.Pointer
	fnFreeVideoResult    unsafe.Pointer
	fnSaveVideoAsGif     unsafe.Pointer
	fnSaveVideoFrames    unsafe.Pointer
)

// lastError retrieves the last error from the Rust binding.
func lastError() string {
	if fnCandleLastError == nil {
		return "library not initialized"
	}
	cStr := C.call_candle_last_error(fnCandleLastError)
	if cStr == nil {
		return "unknown error"
	}
	return C.GoString(cStr)
}

// Version returns the candle binding version string.
func Version() string {
	if !initialized || fnCandleBindingVersion == nil {
		return "unknown"
	}
	cStr := C.call_candle_binding_version(fnCandleBindingVersion)
	if cStr == nil {
		return "unknown"
	}
	return C.GoString(cStr)
}
