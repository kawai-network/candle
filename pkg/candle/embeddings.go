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

// EmbeddingConfig configures an embedding pipeline.
type EmbeddingConfig struct {
	ModelID   string `json:"model_id"`
	CacheDir  string `json:"cache_dir,omitempty"`
	Normalize bool   `json:"normalize"`
}

// EmbeddingPipeline wraps a Rust embedding pipeline.
type EmbeddingPipeline struct {
	ptr *C.EmbeddingPipelineWrapper
}

// NewEmbeddingPipeline creates a new embedding pipeline.
// The model is automatically downloaded from HF Hub if not cached.
func NewEmbeddingPipeline(cfg EmbeddingConfig) (*EmbeddingPipeline, error) {
	if !initialized {
		return nil, errors.New("candle library not initialized")
	}

	configJSON, err := json.Marshal(cfg)
	if err != nil {
		return nil, err
	}

	cConfig := C.CString(string(configJSON))
	defer C.free(unsafe.Pointer(cConfig))

	ptr := C.call_new_embedding_pipeline(fnNewEmbeddingPipeline, cConfig)
	if ptr == nil {
		return nil, errors.New(lastError())
	}

	p := &EmbeddingPipeline{ptr: ptr}
	runtime.SetFinalizer(p, func(obj *EmbeddingPipeline) {
		obj.Close()
	})
	return p, nil
}

// Embed generates an embedding vector for a single text.
func (p *EmbeddingPipeline) Embed(text string) ([]float32, error) {
	if p.ptr == nil {
		return nil, errors.New("pipeline is closed")
	}

	cText := C.CString(text)
	defer C.free(unsafe.Pointer(cText))

	res := C.call_run_embedding(fnRunEmbedding, p.ptr, cText)
	if res == nil {
		return nil, errors.New(lastError())
	}
	defer C.call_free_embedding_result(fnFreeEmbeddingResult, res)

	dim := int(res.dim)
	data := unsafe.Slice((*float32)(unsafe.Pointer(res.data)), dim)
	result := make([]float32, dim)
	copy(result, data)

	return result, nil
}

// EmbedBatch generates embedding vectors for multiple texts.
func (p *EmbeddingPipeline) EmbedBatch(texts []string) ([][]float32, error) {
	if p.ptr == nil {
		return nil, errors.New("pipeline is closed")
	}
	if len(texts) == 0 {
		return nil, errors.New("texts cannot be empty")
	}

	cTexts := make([]*C.char, len(texts))
	for i, t := range texts {
		cTexts[i] = C.CString(t)
		defer C.free(unsafe.Pointer(cTexts[i]))
	}

	res := C.call_run_embedding_batch(
		fnRunEmbeddingBatch,
		p.ptr,
		&cTexts[0],
		C.size_t(len(texts)),
	)
	if res == nil {
		return nil, errors.New(lastError())
	}
	defer C.call_free_batch_embedding_result(fnFreeBatchEmbeddingResult, res)

	dim := int(res.dim)
	count := int(res.count)
	total := dim * count

	flatData := unsafe.Slice((*float32)(unsafe.Pointer(res.data)), total)
	results := make([][]float32, count)
	for i := 0; i < count; i++ {
		results[i] = make([]float32, dim)
		copy(results[i], flatData[i*dim:(i+1)*dim])
	}

	return results, nil
}

// Close frees the underlying Rust resources. Safe to call multiple times.
func (p *EmbeddingPipeline) Close() {
	if p.ptr != nil {
		C.call_free_embedding_pipeline(fnFreeEmbeddingPipeline, p.ptr)
		p.ptr = nil
	}
}
