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

// T5Config configures a T5 seq2seq pipeline.
type T5Config struct {
	// ModelID is the HuggingFace model identifier (e.g. "t5-small", "google/flan-t5-base").
	ModelID string `json:"model_id"`
	// CacheDir overrides the default HF cache directory.
	CacheDir string `json:"cache_dir,omitempty"`
	// Revision overrides the default git revision for the model.
	Revision string `json:"revision,omitempty"`
	// DisableCache disables KV cache (slower but uses less memory).
	DisableCache bool `json:"disable_cache,omitempty"`
}

// T5GenerateOpts provides per-call generation parameters.
type T5GenerateOpts struct {
	// MaxTokens is the maximum number of tokens to generate. Default: 256.
	MaxTokens int `json:"max_tokens,omitempty"`
	// Temperature for sampling (0.0 = greedy). Default: 0.0.
	Temperature float64 `json:"temperature,omitempty"`
	// TopP nucleus sampling probability cutoff.
	TopP float64 `json:"top_p,omitempty"`
	// RepeatPenalty penalizes repeated tokens. 1.0 = no penalty. Default: 1.1.
	RepeatPenalty float64 `json:"repeat_penalty,omitempty"`
}

// T5Pipeline wraps a Rust T5 seq2seq pipeline.
type T5Pipeline struct {
	ptr *C.T5PipelineWrapper
}

// NewT5Pipeline creates a new T5 text-to-text pipeline.
// Supports models like t5-small, t5-base, google/flan-t5-base, etc.
// Use prompts like "translate English to French: ...", "summarize: ...", etc.
func NewT5Pipeline(cfg T5Config) (*T5Pipeline, error) {
	if !initialized {
		return nil, errors.New("candle library not initialized")
	}

	configJSON, err := json.Marshal(cfg)
	if err != nil {
		return nil, err
	}

	cConfig := C.CString(string(configJSON))
	defer C.free(unsafe.Pointer(cConfig))

	ptr := C.call_new_t5_pipeline(fnNewT5Pipeline, cConfig)
	if ptr == nil {
		return nil, errors.New(lastError())
	}

	p := &T5Pipeline{ptr: ptr}
	runtime.SetFinalizer(p, func(obj *T5Pipeline) {
		obj.Close()
	})
	return p, nil
}

// Generate runs seq2seq generation with the given prompt.
// For translation: "translate English to French: The house is wonderful."
// For summarization: "summarize: <long text>"
func (p *T5Pipeline) Generate(prompt string, opts ...T5GenerateOpts) (string, error) {
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

	res := C.call_run_t5_generate(fnRunT5Generate, p.ptr, cPrompt, cParams)
	if res == nil {
		return "", errors.New(lastError())
	}
	defer C.call_free_t5_result(fnFreeT5Result, res)

	return C.GoString(res.text), nil
}

// Close frees the underlying Rust resources. Safe to call multiple times.
func (p *T5Pipeline) Close() {
	if p.ptr != nil {
		C.call_free_t5_pipeline(fnFreeT5Pipeline, p.ptr)
		p.ptr = nil
	}
}
