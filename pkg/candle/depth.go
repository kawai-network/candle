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

// DepthConfig configures a depth estimation pipeline.
type DepthConfig struct {
	// ModelID for the depth head (e.g. "jeroenvlek/depth-anything-v2-safetensors").
	ModelID string `json:"model_id"`
	// DINOv2ModelID for the backbone (default: "lmz/candle-dino-v2").
	DINOv2ModelID string `json:"dinov2_model_id,omitempty"`
	// DINOv2File specific weights file (default: "dinov2_vits14.safetensors").
	DINOv2File string `json:"dinov2_file,omitempty"`
	// DepthFile specific weights file (default: "depth_anything_v2_vits.safetensors").
	DepthFile string `json:"depth_file,omitempty"`
	CacheDir  string `json:"cache_dir,omitempty"`
}

// DepthResult holds the depth estimation output.
type DepthResult struct {
	// Data is a normalized depth map (0.0 = close, 1.0 = far), row-major.
	Data   []float32
	Height int
	Width  int
}

// DepthPipeline wraps a Rust depth estimation pipeline.
type DepthPipeline struct {
	ptr *C.DepthPipelineWrapper
}

// NewDepthPipeline creates a new depth estimation pipeline.
func NewDepthPipeline(cfg DepthConfig) (*DepthPipeline, error) {
	if !initialized {
		return nil, errors.New("candle library not initialized")
	}

	configJSON, err := json.Marshal(cfg)
	if err != nil {
		return nil, err
	}

	cConfig := C.CString(string(configJSON))
	defer C.free(unsafe.Pointer(cConfig))

	ptr := C.call_new_depth_pipeline(fnNewDepthPipeline, cConfig)
	if ptr == nil {
		return nil, errors.New(lastError())
	}

	p := &DepthPipeline{ptr: ptr}
	runtime.SetFinalizer(p, func(obj *DepthPipeline) {
		obj.Close()
	})
	return p, nil
}

// EstimateDepth estimates a depth map from an image file.
func (p *DepthPipeline) EstimateDepth(imagePath string) (*DepthResult, error) {
	if p.ptr == nil {
		return nil, errors.New("pipeline is closed")
	}

	cPath := C.CString(imagePath)
	defer C.free(unsafe.Pointer(cPath))

	res := C.call_run_depth_estimation(fnRunDepthEstimation, p.ptr, cPath)
	if res == nil {
		return nil, errors.New(lastError())
	}
	defer C.call_free_depth_result(fnFreeDepthResult, res)

	h := int(res.height)
	w := int(res.width)
	total := h * w
	data := unsafe.Slice((*float32)(unsafe.Pointer(res.data)), total)
	result := &DepthResult{
		Data:   make([]float32, total),
		Height: h,
		Width:  w,
	}
	copy(result.Data, data)

	return result, nil
}

// Close frees the underlying Rust resources. Safe to call multiple times.
func (p *DepthPipeline) Close() {
	if p.ptr != nil {
		C.call_free_depth_pipeline(fnFreeDepthPipeline, p.ptr)
		p.ptr = nil
	}
}
