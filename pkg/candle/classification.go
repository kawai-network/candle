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

// ClassificationConfig configures an image classification pipeline.
type ClassificationConfig struct {
	ModelID   string `json:"model_id"`
	CacheDir  string `json:"cache_dir,omitempty"`
	NumLabels int    `json:"num_labels,omitempty"`
}

// ClassificationPrediction is a single classification result.
type ClassificationPrediction struct {
	Label string
	Score float32
}

// ClassificationPipeline wraps a Rust image classification pipeline.
type ClassificationPipeline struct {
	ptr *C.ClassificationPipelineWrapper
}

// NewClassificationPipeline creates a new image classification pipeline.
func NewClassificationPipeline(cfg ClassificationConfig) (*ClassificationPipeline, error) {
	if !initialized {
		return nil, errors.New("candle library not initialized")
	}

	configJSON, err := json.Marshal(cfg)
	if err != nil {
		return nil, err
	}

	cConfig := C.CString(string(configJSON))
	defer C.free(unsafe.Pointer(cConfig))

	ptr := C.call_new_classification_pipeline(fnNewClassificationPipeline, cConfig)
	if ptr == nil {
		return nil, errors.New(lastError())
	}

	p := &ClassificationPipeline{ptr: ptr}
	runtime.SetFinalizer(p, func(obj *ClassificationPipeline) {
		obj.Close()
	})
	return p, nil
}

// Classify classifies an image and returns top-k predictions.
func (p *ClassificationPipeline) Classify(imagePath string, topK int) ([]ClassificationPrediction, error) {
	if p.ptr == nil {
		return nil, errors.New("pipeline is closed")
	}

	cPath := C.CString(imagePath)
	defer C.free(unsafe.Pointer(cPath))

	if topK <= 0 {
		topK = 5
	}

	res := C.call_run_classification(fnRunClassification, p.ptr, cPath, C.size_t(topK))
	if res == nil {
		return nil, errors.New(lastError())
	}
	defer C.call_free_classification_result(fnFreeClassificationResult, res)

	count := int(res.count)
	preds := unsafe.Slice(res.predictions, count)
	results := make([]ClassificationPrediction, count)
	for i := 0; i < count; i++ {
		results[i] = ClassificationPrediction{
			Label: C.GoString(preds[i].label),
			Score: float32(preds[i].score),
		}
	}

	return results, nil
}

// Close frees the underlying Rust resources. Safe to call multiple times.
func (p *ClassificationPipeline) Close() {
	if p.ptr != nil {
		C.call_free_classification_pipeline(fnFreeClassificationPipeline, p.ptr)
		p.ptr = nil
	}
}
