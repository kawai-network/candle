#ifndef CANDLE_H
#define CANDLE_H

#include <stdlib.h>
#include <stdint.h>

// --- Core ---

typedef const char* (*candle_last_error_t)();
typedef const char* (*candle_binding_version_t)();

// --- Text Generation ---

typedef struct {
    void* _opaque;
} TextGenerationPipelineWrapper;

typedef struct {
    char* text;
    size_t tokens_generated;
} TextGenerationResult;

typedef TextGenerationPipelineWrapper* (*new_text_generation_pipeline_t)(const char* config_json);
typedef TextGenerationResult* (*run_text_generation_t)(TextGenerationPipelineWrapper* wrapper, const char* prompt, const char* params_json);
typedef void (*free_text_generation_pipeline_t)(TextGenerationPipelineWrapper* wrapper);
typedef void (*free_text_generation_result_t)(TextGenerationResult* result);

// --- Embeddings ---

typedef struct {
    void* _opaque;
} EmbeddingPipelineWrapper;

typedef struct {
    float* data;
    size_t dim;
} EmbeddingResult;

typedef struct {
    float* data;
    size_t dim;
    size_t count;
} BatchEmbeddingResult;

typedef EmbeddingPipelineWrapper* (*new_embedding_pipeline_t)(const char* config_json);
typedef EmbeddingResult* (*run_embedding_t)(EmbeddingPipelineWrapper* wrapper, const char* text);
typedef BatchEmbeddingResult* (*run_embedding_batch_t)(EmbeddingPipelineWrapper* wrapper, const char** texts, size_t count);
typedef void (*free_embedding_pipeline_t)(EmbeddingPipelineWrapper* wrapper);
typedef void (*free_embedding_result_t)(EmbeddingResult* result);
typedef void (*free_batch_embedding_result_t)(BatchEmbeddingResult* result);

// --- Classification ---

typedef struct {
    void* _opaque;
} ClassificationPipelineWrapper;

typedef struct {
    char* label;
    float score;
} ClassificationPrediction;

typedef struct {
    ClassificationPrediction* predictions;
    size_t count;
} ClassificationResult;

typedef ClassificationPipelineWrapper* (*new_classification_pipeline_t)(const char* config_json);
typedef ClassificationResult* (*run_classification_t)(ClassificationPipelineWrapper* wrapper, const char* image_path, size_t top_k);
typedef void (*free_classification_pipeline_t)(ClassificationPipelineWrapper* wrapper);
typedef void (*free_classification_result_t)(ClassificationResult* result);

// --- CLIP ---

typedef struct {
    void* _opaque;
} ClipPipelineWrapper;

typedef struct {
    float* scores;
    size_t count;
} ClipScoreResult;

typedef struct {
    float* data;
    size_t dim;
} ClipEmbeddingResult;

typedef ClipPipelineWrapper* (*new_clip_pipeline_t)(const char* config_json);
typedef ClipScoreResult* (*run_clip_score_t)(ClipPipelineWrapper* wrapper, const char* image_path, const char** texts, size_t text_count);
typedef ClipEmbeddingResult* (*run_clip_embed_image_t)(ClipPipelineWrapper* wrapper, const char* image_path);
typedef ClipEmbeddingResult* (*run_clip_embed_text_t)(ClipPipelineWrapper* wrapper, const char* text);
typedef void (*free_clip_pipeline_t)(ClipPipelineWrapper* wrapper);
typedef void (*free_clip_score_result_t)(ClipScoreResult* result);
typedef void (*free_clip_embedding_result_t)(ClipEmbeddingResult* result);

// --- Depth Estimation ---

typedef struct {
    void* _opaque;
} DepthPipelineWrapper;

typedef struct {
    float* data;
    size_t height;
    size_t width;
} DepthResult;

typedef DepthPipelineWrapper* (*new_depth_pipeline_t)(const char* config_json);
typedef DepthResult* (*run_depth_estimation_t)(DepthPipelineWrapper* wrapper, const char* image_path);
typedef void (*free_depth_pipeline_t)(DepthPipelineWrapper* wrapper);
typedef void (*free_depth_result_t)(DepthResult* result);

// --- Segmentation ---

typedef struct {
    void* _opaque;
} SegmentationPipelineWrapper;

typedef struct {
    int32_t* data;
    size_t height;
    size_t width;
    size_t num_labels;
} SegmentationResult;

typedef SegmentationPipelineWrapper* (*new_segmentation_pipeline_t)(const char* config_json);
typedef SegmentationResult* (*run_segmentation_t)(SegmentationPipelineWrapper* wrapper, const char* image_path);
typedef void (*free_segmentation_pipeline_t)(SegmentationPipelineWrapper* wrapper);
typedef void (*free_segmentation_result_t)(SegmentationResult* result);

// --- Helper call functions ---

// Core
static inline const char* call_candle_last_error(void* f) {
    return ((candle_last_error_t)f)();
}

static inline const char* call_candle_binding_version(void* f) {
    return ((candle_binding_version_t)f)();
}

// Text Generation
static inline TextGenerationPipelineWrapper* call_new_text_generation_pipeline(void* f, const char* config_json) {
    return ((new_text_generation_pipeline_t)f)(config_json);
}

static inline TextGenerationResult* call_run_text_generation(void* f, TextGenerationPipelineWrapper* w, const char* prompt, const char* params_json) {
    return ((run_text_generation_t)f)(w, prompt, params_json);
}

static inline void call_free_text_generation_pipeline(void* f, TextGenerationPipelineWrapper* w) {
    ((free_text_generation_pipeline_t)f)(w);
}

static inline void call_free_text_generation_result(void* f, TextGenerationResult* r) {
    ((free_text_generation_result_t)f)(r);
}

// Embeddings
static inline EmbeddingPipelineWrapper* call_new_embedding_pipeline(void* f, const char* config_json) {
    return ((new_embedding_pipeline_t)f)(config_json);
}

static inline EmbeddingResult* call_run_embedding(void* f, EmbeddingPipelineWrapper* w, const char* text) {
    return ((run_embedding_t)f)(w, text);
}

static inline BatchEmbeddingResult* call_run_embedding_batch(void* f, EmbeddingPipelineWrapper* w, const char** texts, size_t count) {
    return ((run_embedding_batch_t)f)(w, texts, count);
}

static inline void call_free_embedding_pipeline(void* f, EmbeddingPipelineWrapper* w) {
    ((free_embedding_pipeline_t)f)(w);
}

static inline void call_free_embedding_result(void* f, EmbeddingResult* r) {
    ((free_embedding_result_t)f)(r);
}

static inline void call_free_batch_embedding_result(void* f, BatchEmbeddingResult* r) {
    ((free_batch_embedding_result_t)f)(r);
}

// Classification
static inline ClassificationPipelineWrapper* call_new_classification_pipeline(void* f, const char* config_json) {
    return ((new_classification_pipeline_t)f)(config_json);
}

static inline ClassificationResult* call_run_classification(void* f, ClassificationPipelineWrapper* w, const char* image_path, size_t top_k) {
    return ((run_classification_t)f)(w, image_path, top_k);
}

static inline void call_free_classification_pipeline(void* f, ClassificationPipelineWrapper* w) {
    ((free_classification_pipeline_t)f)(w);
}

static inline void call_free_classification_result(void* f, ClassificationResult* r) {
    ((free_classification_result_t)f)(r);
}

// CLIP
static inline ClipPipelineWrapper* call_new_clip_pipeline(void* f, const char* config_json) {
    return ((new_clip_pipeline_t)f)(config_json);
}

static inline ClipScoreResult* call_run_clip_score(void* f, ClipPipelineWrapper* w, const char* image_path, const char** texts, size_t text_count) {
    return ((run_clip_score_t)f)(w, image_path, texts, text_count);
}

static inline ClipEmbeddingResult* call_run_clip_embed_image(void* f, ClipPipelineWrapper* w, const char* image_path) {
    return ((run_clip_embed_image_t)f)(w, image_path);
}

static inline ClipEmbeddingResult* call_run_clip_embed_text(void* f, ClipPipelineWrapper* w, const char* text) {
    return ((run_clip_embed_text_t)f)(w, text);
}

static inline void call_free_clip_pipeline(void* f, ClipPipelineWrapper* w) {
    ((free_clip_pipeline_t)f)(w);
}

static inline void call_free_clip_score_result(void* f, ClipScoreResult* r) {
    ((free_clip_score_result_t)f)(r);
}

static inline void call_free_clip_embedding_result(void* f, ClipEmbeddingResult* r) {
    ((free_clip_embedding_result_t)f)(r);
}

// Depth Estimation
static inline DepthPipelineWrapper* call_new_depth_pipeline(void* f, const char* config_json) {
    return ((new_depth_pipeline_t)f)(config_json);
}

static inline DepthResult* call_run_depth_estimation(void* f, DepthPipelineWrapper* w, const char* image_path) {
    return ((run_depth_estimation_t)f)(w, image_path);
}

static inline void call_free_depth_pipeline(void* f, DepthPipelineWrapper* w) {
    ((free_depth_pipeline_t)f)(w);
}

static inline void call_free_depth_result(void* f, DepthResult* r) {
    ((free_depth_result_t)f)(r);
}

// Segmentation
static inline SegmentationPipelineWrapper* call_new_segmentation_pipeline(void* f, const char* config_json) {
    return ((new_segmentation_pipeline_t)f)(config_json);
}

static inline SegmentationResult* call_run_segmentation(void* f, SegmentationPipelineWrapper* w, const char* image_path) {
    return ((run_segmentation_t)f)(w, image_path);
}

static inline void call_free_segmentation_pipeline(void* f, SegmentationPipelineWrapper* w) {
    ((free_segmentation_pipeline_t)f)(w);
}

static inline void call_free_segmentation_result(void* f, SegmentationResult* r) {
    ((free_segmentation_result_t)f)(r);
}

// --- Whisper ---

typedef struct {
    void* _opaque;
} WhisperPipelineWrapper;

typedef struct {
    char* text;
    double start;
    double end;
} WhisperSegment;

typedef struct {
    char* text;
    WhisperSegment* segments;
    size_t segment_count;
} WhisperResult;

typedef WhisperPipelineWrapper* (*new_whisper_pipeline_t)(const char* config_json);
typedef WhisperResult* (*run_whisper_transcribe_t)(WhisperPipelineWrapper* wrapper, const char* audio_path, const char* params_json);
typedef void (*free_whisper_pipeline_t)(WhisperPipelineWrapper* wrapper);
typedef void (*free_whisper_result_t)(WhisperResult* result);

// Whisper helpers
static inline WhisperPipelineWrapper* call_new_whisper_pipeline(void* f, const char* config_json) {
    return ((new_whisper_pipeline_t)f)(config_json);
}

static inline WhisperResult* call_run_whisper_transcribe(void* f, WhisperPipelineWrapper* w, const char* audio_path, const char* params_json) {
    return ((run_whisper_transcribe_t)f)(w, audio_path, params_json);
}

static inline void call_free_whisper_pipeline(void* f, WhisperPipelineWrapper* w) {
    ((free_whisper_pipeline_t)f)(w);
}

static inline void call_free_whisper_result(void* f, WhisperResult* r) {
    ((free_whisper_result_t)f)(r);
}

// --- T5 ---

typedef struct {
    void* _opaque;
} T5PipelineWrapper;

typedef struct {
    char* text;
} T5Result;

typedef T5PipelineWrapper* (*new_t5_pipeline_t)(const char* config_json);
typedef T5Result* (*run_t5_generate_t)(T5PipelineWrapper* wrapper, const char* prompt, const char* params_json);
typedef void (*free_t5_pipeline_t)(T5PipelineWrapper* wrapper);
typedef void (*free_t5_result_t)(T5Result* result);

// T5 helpers
static inline T5PipelineWrapper* call_new_t5_pipeline(void* f, const char* config_json) {
    return ((new_t5_pipeline_t)f)(config_json);
}

static inline T5Result* call_run_t5_generate(void* f, T5PipelineWrapper* w, const char* prompt, const char* params_json) {
    return ((run_t5_generate_t)f)(w, prompt, params_json);
}

static inline void call_free_t5_pipeline(void* f, T5PipelineWrapper* w) {
    ((free_t5_pipeline_t)f)(w);
}

static inline void call_free_t5_result(void* f, T5Result* r) {
    ((free_t5_result_t)f)(r);
}

// --- Translation (Marian MT) ---

typedef struct {
    void* _opaque;
} TranslationPipelineWrapper;

typedef struct {
    char* text;
} TranslationResult;

typedef TranslationPipelineWrapper* (*new_translation_pipeline_t)(const char* config_json);
typedef TranslationResult* (*run_translation_t)(TranslationPipelineWrapper* wrapper, const char* text, const char* params_json);
typedef void (*free_translation_pipeline_t)(TranslationPipelineWrapper* wrapper);
typedef void (*free_translation_result_t)(TranslationResult* result);

// Translation helpers
static inline TranslationPipelineWrapper* call_new_translation_pipeline(void* f, const char* config_json) {
    return ((new_translation_pipeline_t)f)(config_json);
}

static inline TranslationResult* call_run_translation(void* f, TranslationPipelineWrapper* w, const char* text, const char* params_json) {
    return ((run_translation_t)f)(w, text, params_json);
}

static inline void call_free_translation_pipeline(void* f, TranslationPipelineWrapper* w) {
    ((free_translation_pipeline_t)f)(w);
}

static inline void call_free_translation_result(void* f, TranslationResult* r) {
    ((free_translation_result_t)f)(r);
}

// --- Video Generation ---

typedef struct {
    void* _opaque;
} VideoPipelineWrapper;

typedef struct {
    float* data;
    size_t height;
    size_t width;
    size_t channels;
} VideoFrame;

typedef struct {
    VideoFrame* frames;
    size_t frame_count;
    size_t fps;
    char* error;
} VideoResult;

typedef VideoPipelineWrapper* (*new_video_pipeline_t)(const char* config_json);
typedef VideoResult* (*run_video_generation_t)(VideoPipelineWrapper* wrapper, const char* prompt, const char* params_json);
typedef void (*free_video_pipeline_t)(VideoPipelineWrapper* wrapper);
typedef void (*free_video_result_t)(VideoResult* result);
typedef int (*save_video_as_gif_t)(const VideoResult* result, const char* output_path);
typedef int (*save_video_frames_t)(const VideoResult* result, const char* output_dir);

// Video helpers
static inline VideoPipelineWrapper* call_new_video_pipeline(void* f, const char* config_json) {
    return ((new_video_pipeline_t)f)(config_json);
}

static inline VideoResult* call_run_video_generation(void* f, VideoPipelineWrapper* w, const char* prompt, const char* params_json) {
    return ((run_video_generation_t)f)(w, prompt, params_json);
}

static inline void call_free_video_pipeline(void* f, VideoPipelineWrapper* w) {
    ((free_video_pipeline_t)f)(w);
}

static inline void call_free_video_result(void* f, VideoResult* r) {
    ((free_video_result_t)f)(r);
}

static inline int call_save_video_as_gif(void* f, const VideoResult* r, const char* output_path) {
    return ((save_video_as_gif_t)f)(r, output_path);
}

static inline int call_save_video_frames(void* f, const VideoResult* r, const char* output_dir) {
    return ((save_video_frames_t)f)(r, output_dir);
}

#endif /* CANDLE_H */
