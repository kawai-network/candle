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

// SegmentationConfig configures a semantic segmentation pipeline.
type SegmentationConfig struct {
	ModelID   string `json:"model_id"`
	CacheDir  string `json:"cache_dir,omitempty"`
	NumLabels int    `json:"num_labels,omitempty"`
}

// SegmentationResult holds the semantic segmentation output.
type SegmentationResult struct {
	// Data is a class ID map (one int32 per pixel), row-major.
	Data      []int32
	Height    int
	Width     int
	NumLabels int
}

// SegmentationPipeline wraps a Rust semantic segmentation pipeline.
type SegmentationPipeline struct {
	ptr *C.SegmentationPipelineWrapper
}

// NewSegmentationPipeline creates a new semantic segmentation pipeline.
func NewSegmentationPipeline(cfg SegmentationConfig) (*SegmentationPipeline, error) {
	if !initialized {
		return nil, errors.New("candle library not initialized")
	}

	configJSON, err := json.Marshal(cfg)
	if err != nil {
		return nil, err
	}

	cConfig := C.CString(string(configJSON))
	defer C.free(unsafe.Pointer(cConfig))

	ptr := C.call_new_segmentation_pipeline(fnNewSegmentationPipeline, cConfig)
	if ptr == nil {
		return nil, errors.New(lastError())
	}

	p := &SegmentationPipeline{ptr: ptr}
	runtime.SetFinalizer(p, func(obj *SegmentationPipeline) {
		obj.Close()
	})
	return p, nil
}

// Segment performs semantic segmentation on an image file.
func (p *SegmentationPipeline) Segment(imagePath string) (*SegmentationResult, error) {
	if p.ptr == nil {
		return nil, errors.New("pipeline is closed")
	}

	cPath := C.CString(imagePath)
	defer C.free(unsafe.Pointer(cPath))

	res := C.call_run_segmentation(fnRunSegmentation, p.ptr, cPath)
	if res == nil {
		return nil, errors.New(lastError())
	}
	defer C.call_free_segmentation_result(fnFreeSegmentationResult, res)

	h := int(res.height)
	w := int(res.width)
	total := h * w
	data := unsafe.Slice((*int32)(unsafe.Pointer(res.data)), total)
	result := &SegmentationResult{
		Data:      make([]int32, total),
		Height:    h,
		Width:     w,
		NumLabels: int(res.num_labels),
	}
	copy(result.Data, data)

	return result, nil
}

// Close frees the underlying Rust resources. Safe to call multiple times.
func (p *SegmentationPipeline) Close() {
	if p.ptr != nil {
		C.call_free_segmentation_pipeline(fnFreeSegmentationPipeline, p.ptr)
		p.ptr = nil
	}
}
