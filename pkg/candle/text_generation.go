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

// TextGenerationConfig configures a text generation pipeline.
type TextGenerationConfig struct {
	ModelID       string  `json:"model_id"`
	CacheDir      string  `json:"cache_dir,omitempty"`
	Device        string  `json:"device,omitempty"`
	MaxTokens     int     `json:"max_tokens,omitempty"`
	Temperature   float64 `json:"temperature,omitempty"`
	TopP          float64 `json:"top_p,omitempty"`
	RepeatPenalty float64 `json:"repeat_penalty,omitempty"`
}

// GenerateOpts provides per-call generation parameters.
type GenerateOpts struct {
	MaxTokens     int     `json:"max_tokens,omitempty"`
	Temperature   float64 `json:"temperature,omitempty"`
	TopP          float64 `json:"top_p,omitempty"`
	RepeatPenalty float64 `json:"repeat_penalty,omitempty"`
	Seed          uint64  `json:"seed,omitempty"`
}

// TextGenerationPipeline wraps a Rust text generation pipeline.
type TextGenerationPipeline struct {
	ptr *C.TextGenerationPipelineWrapper
}

// NewTextGenerationPipeline creates a new text generation pipeline.
// The model is automatically downloaded from HF Hub if not cached.
func NewTextGenerationPipeline(cfg TextGenerationConfig) (*TextGenerationPipeline, error) {
	if !initialized {
		return nil, errors.New("candle library not initialized")
	}

	configJSON, err := json.Marshal(cfg)
	if err != nil {
		return nil, err
	}

	cConfig := C.CString(string(configJSON))
	defer C.free(unsafe.Pointer(cConfig))

	ptr := C.call_new_text_generation_pipeline(fnNewTextGenerationPipeline, cConfig)
	if ptr == nil {
		return nil, errors.New(lastError())
	}

	p := &TextGenerationPipeline{ptr: ptr}
	runtime.SetFinalizer(p, func(obj *TextGenerationPipeline) {
		obj.Close()
	})
	return p, nil
}

// Generate produces text continuation for the given prompt.
func (p *TextGenerationPipeline) Generate(prompt string, opts ...GenerateOpts) (string, error) {
	if p.ptr == nil {
		return "", errors.New("pipeline is closed")
	}

	cPrompt := C.CString(prompt)
	defer C.free(unsafe.Pointer(cPrompt))

	var cParams *C.char
	if len(opts) > 0 {
		paramsJSON, err := json.Marshal(opts[0])
		if err != nil {
			return "", err
		}
		cParams = C.CString(string(paramsJSON))
		defer C.free(unsafe.Pointer(cParams))
	}

	res := C.call_run_text_generation(fnRunTextGeneration, p.ptr, cPrompt, cParams)
	if res == nil {
		return "", errors.New(lastError())
	}
	defer C.call_free_text_generation_result(fnFreeTextGenerationResult, res)

	return C.GoString(res.text), nil
}

// Close frees the underlying Rust resources. Safe to call multiple times.
func (p *TextGenerationPipeline) Close() {
	if p.ptr != nil {
		C.call_free_text_generation_pipeline(fnFreeTextGenerationPipeline, p.ptr)
		p.ptr = nil
	}
}
