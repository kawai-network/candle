package candle

/*
#include "candle.h"
*/
import "C"
import (
	"encoding/json"
	"errors"
	"fmt"
	"runtime"
	"unsafe"
)

type VideoConfig struct {
	ModelID        string `json:"model_id"`
	CacheDir       string `json:"cache_dir,omitempty"`
	LibraryVersion string `json:"library_version,omitempty"`
}

type VideoGenerationParams struct {
	Height            int     `json:"height,omitempty"`
	Width             int     `json:"width,omitempty"`
	NumFrames         int     `json:"num_frames,omitempty"`
	NumInferenceSteps int     `json:"num_inference_steps,omitempty"`
	GuidanceScale     float32 `json:"guidance_scale,omitempty"`
	FrameRate         int     `json:"frame_rate,omitempty"`
	Seed              uint64  `json:"seed,omitempty"`
}

type VideoFrame struct {
	Data     []float32
	Height   int
	Width    int
	Channels int
}

type VideoResult struct {
	Frames     []VideoFrame
	FrameCount int
	FPS        int

	cPtr *C.VideoResult
}

type VideoPipeline struct {
	ptr *C.VideoPipelineWrapper
}

func NewVideoPipeline(cfg VideoConfig) (*VideoPipeline, error) {
	if !initialized {
		return nil, errors.New("candle library not initialized")
	}

	if !videoAvailable || fnNewVideoPipeline == nil {
		if err := initVideoLibrary(cfg.LibraryVersion); err != nil {
			return nil, fmt.Errorf("video pipeline not available: %w (set CANDLE_LIB_PATH or the library will be downloaded automatically)", err)
		}
	}

	configJSON, err := json.Marshal(cfg)
	if err != nil {
		return nil, err
	}

	cConfig := C.CString(string(configJSON))
	defer C.free(unsafe.Pointer(cConfig))

	ptr := C.call_new_video_pipeline(fnNewVideoPipeline, cConfig)
	if ptr == nil {
		return nil, errors.New(lastError())
	}

	p := &VideoPipeline{ptr: ptr}
	runtime.SetFinalizer(p, func(obj *VideoPipeline) {
		obj.Close()
	})
	return p, nil
}

func (p *VideoPipeline) Generate(prompt string, params VideoGenerationParams) (*VideoResult, error) {
	if p.ptr == nil {
		return nil, errors.New("pipeline is closed")
	}

	cPrompt := C.CString(prompt)
	defer C.free(unsafe.Pointer(cPrompt))

	paramsJSON, err := json.Marshal(params)
	if err != nil {
		return nil, err
	}

	cParams := C.CString(string(paramsJSON))
	defer C.free(unsafe.Pointer(cParams))

	res := C.call_run_video_generation(fnRunVideoGeneration, p.ptr, cPrompt, cParams)
	if res == nil {
		return nil, errors.New(lastError())
	}

	if res.error != nil {
		errMsg := C.GoString(res.error)
		C.call_free_video_result(fnFreeVideoResult, res)
		return nil, errors.New(errMsg)
	}

	if res.frame_count == 0 || res.frames == nil {
		C.call_free_video_result(fnFreeVideoResult, res)
		return nil, errors.New("no frames generated")
	}

	frames := unsafe.Slice(res.frames, res.frame_count)
	result := &VideoResult{
		Frames:     make([]VideoFrame, res.frame_count),
		FrameCount: int(res.frame_count),
		FPS:        int(res.fps),
		cPtr:       res,
	}

	for i, frame := range frames {
		if frame.data == nil {
			continue
		}

		dataSize := int(frame.height * frame.width * frame.channels)
		data := unsafe.Slice(frame.data, dataSize)

		frameData := make([]float32, dataSize)
		for j := range data {
			frameData[j] = float32(data[j])
		}

		result.Frames[i] = VideoFrame{
			Data:     frameData,
			Height:   int(frame.height),
			Width:    int(frame.width),
			Channels: int(frame.channels),
		}
	}

	runtime.SetFinalizer(result, func(obj *VideoResult) {
		obj.Close()
	})
	return result, nil
}

func (p *VideoPipeline) Close() {
	if p.ptr != nil {
		C.call_free_video_pipeline(fnFreeVideoPipeline, p.ptr)
		p.ptr = nil
	}
}

func (r *VideoResult) Close() {
	if r.cPtr != nil {
		C.call_free_video_result(fnFreeVideoResult, r.cPtr)
		r.cPtr = nil
	}
}

func (r *VideoResult) SaveGIF(outputPath string) error {
	if r.cPtr == nil {
		return errors.New("video result is already closed or data not available for C export")
	}

	cPath := C.CString(outputPath)
	defer C.free(unsafe.Pointer(cPath))

	ret := C.call_save_video_as_gif(fnSaveVideoAsGif, r.cPtr, cPath)
	if ret != 0 {
		return errors.New("failed to save video as GIF")
	}
	return nil
}

func (r *VideoResult) SaveFrames(outputDir string) error {
	if r.cPtr == nil {
		return errors.New("video result is already closed or data not available for C export")
	}

	cDir := C.CString(outputDir)
	defer C.free(unsafe.Pointer(cDir))

	ret := C.call_save_video_frames(fnSaveVideoFrames, r.cPtr, cDir)
	if ret != 0 {
		return errors.New("failed to save video frames")
	}
	return nil
}

func IsVideoAvailable() bool {
	return videoAvailable
}
