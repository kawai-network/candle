package candle

/*
#include "candle.h"
*/
import "C"
import (
	"encoding/json"
	"errors"
	"runtime"
	"unsafe"
)

// ClipConfig configures a CLIP pipeline.
type ClipConfig struct {
	ModelID  string `json:"model_id"`
	CacheDir string `json:"cache_dir,omitempty"`
}

// ClipPipeline wraps a Rust CLIP model pipeline.
type ClipPipeline struct {
	ptr *C.ClipPipelineWrapper
}

// NewClipPipeline creates a new CLIP pipeline.
func NewClipPipeline(cfg ClipConfig) (*ClipPipeline, error) {
	if !initialized {
		return nil, errors.New("candle library not initialized")
	}

	configJSON, err := json.Marshal(cfg)
	if err != nil {
		return nil, err
	}

	cConfig := C.CString(string(configJSON))
	defer C.free(unsafe.Pointer(cConfig))

	ptr := C.call_new_clip_pipeline(fnNewClipPipeline, cConfig)
	if ptr == nil {
		return nil, errors.New(lastError())
	}

	p := &ClipPipeline{ptr: ptr}
	runtime.SetFinalizer(p, func(obj *ClipPipeline) {
		obj.Close()
	})
	return p, nil
}

// Score computes image-text similarity scores for an image against multiple texts.
// Returns softmax-normalized scores (one per text).
func (p *ClipPipeline) Score(imagePath string, texts []string) ([]float32, error) {
	if p.ptr == nil {
		return nil, errors.New("pipeline is closed")
	}
	if len(texts) == 0 {
		return nil, errors.New("texts cannot be empty")
	}

	cPath := C.CString(imagePath)
	defer C.free(unsafe.Pointer(cPath))

	cTexts := make([]*C.char, len(texts))
	for i, t := range texts {
		cTexts[i] = C.CString(t)
		defer C.free(unsafe.Pointer(cTexts[i]))
	}

	res := C.call_run_clip_score(fnRunClipScore, p.ptr, cPath, &cTexts[0], C.size_t(len(texts)))
	if res == nil {
		return nil, errors.New(lastError())
	}
	defer C.call_free_clip_score_result(fnFreeClipScoreResult, res)

	count := int(res.count)
	data := unsafe.Slice((*float32)(unsafe.Pointer(res.scores)), count)
	result := make([]float32, count)
	copy(result, data)

	return result, nil
}

// EmbedImage generates a CLIP embedding vector for an image.
func (p *ClipPipeline) EmbedImage(imagePath string) ([]float32, error) {
	if p.ptr == nil {
		return nil, errors.New("pipeline is closed")
	}

	cPath := C.CString(imagePath)
	defer C.free(unsafe.Pointer(cPath))

	res := C.call_run_clip_embed_image(fnRunClipEmbedImage, p.ptr, cPath)
	if res == nil {
		return nil, errors.New(lastError())
	}
	defer C.call_free_clip_embedding_result(fnFreeClipEmbeddingResult, res)

	dim := int(res.dim)
	data := unsafe.Slice((*float32)(unsafe.Pointer(res.data)), dim)
	result := make([]float32, dim)
	copy(result, data)

	return result, nil
}

// EmbedText generates a CLIP embedding vector for text.
func (p *ClipPipeline) EmbedText(text string) ([]float32, error) {
	if p.ptr == nil {
		return nil, errors.New("pipeline is closed")
	}

	cText := C.CString(text)
	defer C.free(unsafe.Pointer(cText))

	res := C.call_run_clip_embed_text(fnRunClipEmbedText, p.ptr, cText)
	if res == nil {
		return nil, errors.New(lastError())
	}
	defer C.call_free_clip_embedding_result(fnFreeClipEmbeddingResult, res)

	dim := int(res.dim)
	data := unsafe.Slice((*float32)(unsafe.Pointer(res.data)), dim)
	result := make([]float32, dim)
	copy(result, data)

	return result, nil
}

// Close frees the underlying Rust resources. Safe to call multiple times.
func (p *ClipPipeline) Close() {
	if p.ptr != nil {
		C.call_free_clip_pipeline(fnFreeClipPipeline, p.ptr)
		p.ptr = nil
	}
}
