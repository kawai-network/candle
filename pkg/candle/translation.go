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

// TranslationConfig configures a Marian MT translation pipeline.
type TranslationConfig struct {
	// LanguagePair specifies the translation direction.
	// Supported: "fr-en", "en-fr", "en-es", "en-zh", "en-ru", "en-hi".
	LanguagePair string `json:"language_pair"`
	// CacheDir overrides the default HF cache directory.
	CacheDir string `json:"cache_dir,omitempty"`
}

// TranslateOpts provides per-call translation parameters.
type TranslateOpts struct {
	// MaxTokens is the maximum number of tokens to generate. Default: 512.
	MaxTokens int `json:"max_tokens,omitempty"`
}

// TranslationPipeline wraps a Rust Marian MT translation pipeline.
type TranslationPipeline struct {
	ptr *C.TranslationPipelineWrapper
}

// NewTranslationPipeline creates a new Marian MT translation pipeline.
// The model and tokenizers are automatically downloaded from HF Hub.
func NewTranslationPipeline(cfg TranslationConfig) (*TranslationPipeline, error) {
	if !initialized {
		return nil, errors.New("candle library not initialized")
	}

	configJSON, err := json.Marshal(cfg)
	if err != nil {
		return nil, err
	}

	cConfig := C.CString(string(configJSON))
	defer C.free(unsafe.Pointer(cConfig))

	ptr := C.call_new_translation_pipeline(fnNewTranslationPipeline, cConfig)
	if ptr == nil {
		return nil, errors.New(lastError())
	}

	p := &TranslationPipeline{ptr: ptr}
	runtime.SetFinalizer(p, func(obj *TranslationPipeline) {
		obj.Close()
	})
	return p, nil
}

// Translate translates the given text using the configured language pair.
func (p *TranslationPipeline) Translate(text string, opts ...TranslateOpts) (string, error) {
	if p.ptr == nil {
		return "", errors.New("pipeline is closed")
	}

	cText := C.CString(text)
	defer C.free(unsafe.Pointer(cText))

	var cParams *C.char
	if len(opts) > 0 {
		paramsJSON, err := json.Marshal(opts[0])
		if err != nil {
			return "", err
		}
		cParams = C.CString(string(paramsJSON))
		defer C.free(unsafe.Pointer(cParams))
	}

	res := C.call_run_translation(fnRunTranslation, p.ptr, cText, cParams)
	if res == nil {
		return "", errors.New(lastError())
	}
	defer C.call_free_translation_result(fnFreeTranslationResult, res)

	return C.GoString(res.text), nil
}

// Close frees the underlying Rust resources. Safe to call multiple times.
func (p *TranslationPipeline) Close() {
	if p.ptr != nil {
		C.call_free_translation_pipeline(fnFreeTranslationPipeline, p.ptr)
		p.ptr = nil
	}
}
