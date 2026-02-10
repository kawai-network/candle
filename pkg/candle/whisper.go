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

// WhisperConfig configures a Whisper speech recognition pipeline.
type WhisperConfig struct {
	ModelID  string `json:"model_id"`
	CacheDir string `json:"cache_dir,omitempty"`
}

// TranscribeOpts provides per-call transcription parameters.
type TranscribeOpts struct {
	// Temperature for sampling (0.0 = greedy).
	Temperature float64 `json:"temperature,omitempty"`
	// Language code (e.g. "en", "fr", "de"). Default: "en".
	Language string `json:"language,omitempty"`
}

// WhisperSegment represents a timestamped segment of transcription.
type WhisperSegment struct {
	Text  string
	Start float64
	End   float64
}

// TranscribeResult holds the full transcription output.
type TranscribeResult struct {
	Text     string
	Segments []WhisperSegment
}

// WhisperPipeline wraps a Rust Whisper pipeline.
type WhisperPipeline struct {
	ptr *C.WhisperPipelineWrapper
}

// NewWhisperPipeline creates a new Whisper speech recognition pipeline.
// The model is automatically downloaded from HF Hub.
func NewWhisperPipeline(cfg WhisperConfig) (*WhisperPipeline, error) {
	if !initialized {
		return nil, errors.New("candle library not initialized")
	}

	configJSON, err := json.Marshal(cfg)
	if err != nil {
		return nil, err
	}

	cConfig := C.CString(string(configJSON))
	defer C.free(unsafe.Pointer(cConfig))

	ptr := C.call_new_whisper_pipeline(fnNewWhisperPipeline, cConfig)
	if ptr == nil {
		return nil, errors.New(lastError())
	}

	p := &WhisperPipeline{ptr: ptr}
	runtime.SetFinalizer(p, func(obj *WhisperPipeline) {
		obj.Close()
	})
	return p, nil
}

// Transcribe transcribes a WAV audio file.
// The audio must be 16kHz sample rate.
func (p *WhisperPipeline) Transcribe(audioPath string, opts ...TranscribeOpts) (*TranscribeResult, error) {
	if p.ptr == nil {
		return nil, errors.New("pipeline is closed")
	}

	cPath := C.CString(audioPath)
	defer C.free(unsafe.Pointer(cPath))

	var cParams *C.char
	if len(opts) > 0 {
		paramsJSON, err := json.Marshal(opts[0])
		if err != nil {
			return nil, err
		}
		cParams = C.CString(string(paramsJSON))
		defer C.free(unsafe.Pointer(cParams))
	}

	res := C.call_run_whisper_transcribe(fnRunWhisperTranscribe, p.ptr, cPath, cParams)
	if res == nil {
		return nil, errors.New(lastError())
	}
	defer C.call_free_whisper_result(fnFreeWhisperResult, res)

	result := &TranscribeResult{
		Text: C.GoString(res.text),
	}

	segCount := int(res.segment_count)
	if segCount > 0 {
		segs := unsafe.Slice(res.segments, segCount)
		result.Segments = make([]WhisperSegment, segCount)
		for i := 0; i < segCount; i++ {
			result.Segments[i] = WhisperSegment{
				Text:  C.GoString(segs[i].text),
				Start: float64(segs[i].start),
				End:   float64(segs[i].end),
			}
		}
	}

	return result, nil
}

// Close frees the underlying Rust resources. Safe to call multiple times.
func (p *WhisperPipeline) Close() {
	if p.ptr != nil {
		C.call_free_whisper_pipeline(fnFreeWhisperPipeline, p.ptr)
		p.ptr = nil
	}
}
