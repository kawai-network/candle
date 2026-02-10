use std::ffi::{CStr, CString};
use std::os::raw::c_char;

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;

use crate::{json_str, parse_config_json, set_last_error, create_hf_repo_with_revision, load_weight_files};
use crate::image_utils;

pub struct ClipPipelineWrapper {
    model: candle_transformers::models::clip::ClipModel,
    tokenizer: tokenizers::Tokenizer,
    device: Device,
    image_size: usize,
}

/// Score result from CLIP image-text similarity.
#[repr(C)]
pub struct ClipScoreResult {
    pub scores: *mut f32,
    pub count: usize,
}

/// Embedding result (image or text).
#[repr(C)]
pub struct ClipEmbeddingResult {
    pub data: *mut f32,
    pub dim: usize,
}

fn parse_clip_config(
    config_data: &serde_json::Value,
) -> anyhow::Result<candle_transformers::models::clip::ClipConfig> {
    use candle_transformers::models::clip::text_model::ClipTextConfig;
    use candle_transformers::models::clip::vision_model::ClipVisionConfig;
    use candle_transformers::models::clip::text_model::Activation;

    let tc = config_data
        .get("text_config")
        .ok_or_else(|| anyhow::anyhow!("missing text_config in model config"))?;
    let vc = config_data
        .get("vision_config")
        .ok_or_else(|| anyhow::anyhow!("missing vision_config in model config"))?;

    let text_config = ClipTextConfig {
        vocab_size: tc
            .get("vocab_size")
            .and_then(|v| v.as_u64())
            .unwrap_or(49408) as usize,
        embed_dim: tc
            .get("hidden_size")
            .and_then(|v| v.as_u64())
            .unwrap_or(512) as usize,
        intermediate_size: tc
            .get("intermediate_size")
            .and_then(|v| v.as_u64())
            .unwrap_or(2048) as usize,
        max_position_embeddings: tc
            .get("max_position_embeddings")
            .and_then(|v| v.as_u64())
            .unwrap_or(77) as usize,
        pad_with: None,
        num_hidden_layers: tc
            .get("num_hidden_layers")
            .and_then(|v| v.as_u64())
            .unwrap_or(12) as usize,
        num_attention_heads: tc
            .get("num_attention_heads")
            .and_then(|v| v.as_u64())
            .unwrap_or(8) as usize,
        projection_dim: tc
            .get("projection_dim")
            .or_else(|| config_data.get("projection_dim"))
            .and_then(|v| v.as_u64())
            .unwrap_or(512) as usize,
        activation: Activation::QuickGelu,
    };

    let image_size = vc
        .get("image_size")
        .and_then(|v| v.as_u64())
        .unwrap_or(224) as usize;

    let vision_config = ClipVisionConfig {
        embed_dim: vc
            .get("hidden_size")
            .and_then(|v| v.as_u64())
            .unwrap_or(768) as usize,
        activation: Activation::QuickGelu,
        intermediate_size: vc
            .get("intermediate_size")
            .and_then(|v| v.as_u64())
            .unwrap_or(3072) as usize,
        num_hidden_layers: vc
            .get("num_hidden_layers")
            .and_then(|v| v.as_u64())
            .unwrap_or(12) as usize,
        num_attention_heads: vc
            .get("num_attention_heads")
            .and_then(|v| v.as_u64())
            .unwrap_or(12) as usize,
        projection_dim: vc
            .get("projection_dim")
            .or_else(|| config_data.get("projection_dim"))
            .and_then(|v| v.as_u64())
            .unwrap_or(512) as usize,
        num_channels: vc
            .get("num_channels")
            .and_then(|v| v.as_u64())
            .unwrap_or(3) as usize,
        image_size,
        patch_size: vc
            .get("patch_size")
            .and_then(|v| v.as_u64())
            .unwrap_or(32) as usize,
    };

    Ok(candle_transformers::models::clip::ClipConfig {
        text_config,
        vision_config,
        logit_scale_init_value: config_data
            .get("logit_scale_init_value")
            .and_then(|v| v.as_f64())
            .unwrap_or(2.6592) as f32,
        image_size,
    })
}

fn tokenize_clip(
    tokenizer: &tokenizers::Tokenizer,
    text: &str,
    max_len: usize,
    device: &Device,
) -> anyhow::Result<Tensor> {
    let encoding = tokenizer
        .encode(text, true)
        .map_err(|e| anyhow::anyhow!("tokenization error: {e}"))?;
    let mut ids = encoding.get_ids().to_vec();
    ids.truncate(max_len);
    // Pad to max_len
    while ids.len() < max_len {
        ids.push(0);
    }
    Ok(Tensor::new(&ids[..], device)?.unsqueeze(0)?)
}

#[no_mangle]
pub extern "C" fn new_clip_pipeline(
    config_json: *const c_char,
) -> *mut ClipPipelineWrapper {
    let config = match parse_config_json(config_json) {
        Ok(c) => c,
        Err(e) => {
            set_last_error(e);
            return std::ptr::null_mut();
        }
    };

    let device = Device::Cpu;

    match load_clip_model(&config, &device) {
        Ok(wrapper) => Box::into_raw(Box::new(wrapper)),
        Err(e) => {
            set_last_error(format!("failed to load CLIP model: {e}"));
            std::ptr::null_mut()
        }
    }
}

fn load_clip_model(
    config: &serde_json::Value,
    device: &Device,
) -> anyhow::Result<ClipPipelineWrapper> {
    let model_id = json_str(config, "model_id", "");
    if model_id.is_empty() {
        anyhow::bail!("model_id is required");
    }

    let cache_dir = config
        .get("cache_dir")
        .and_then(|v| v.as_str());

    // For openai CLIP models, safetensors are on refs/pr/15
    let revision = config
        .get("revision")
        .and_then(|v| v.as_str())
        .or_else(|| {
            if model_id.starts_with("openai/clip-") {
                Some("refs/pr/15")
            } else {
                None
            }
        });

    let repo = create_hf_repo_with_revision(model_id, cache_dir, revision)?;

    let config_path = repo.get("config.json")?;
    let config_data: serde_json::Value =
        serde_json::from_reader(std::fs::File::open(&config_path)?)?;

    let clip_config = parse_clip_config(&config_data)?;
    let image_size = clip_config.image_size;

    let tokenizer_path = repo.get("tokenizer.json")?;
    let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow::anyhow!("failed to load tokenizer: {e}"))?;

    let weight_files = load_weight_files(&repo)?;
    let dtype = DType::F32;
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&weight_files, dtype, device)? };

    let model = candle_transformers::models::clip::ClipModel::new(vb, &clip_config)?;

    Ok(ClipPipelineWrapper {
        model,
        tokenizer,
        device: device.clone(),
        image_size,
    })
}

#[no_mangle]
pub extern "C" fn run_clip_score(
    wrapper: *mut ClipPipelineWrapper,
    image_path: *const c_char,
    texts: *const *const c_char,
    text_count: usize,
) -> *mut ClipScoreResult {
    if wrapper.is_null() || image_path.is_null() || texts.is_null() || text_count == 0 {
        set_last_error("null pointer or zero count".to_string());
        return std::ptr::null_mut();
    }

    let wrapper = unsafe { &*wrapper };
    let path_str = unsafe { CStr::from_ptr(image_path) }
        .to_str()
        .unwrap_or_default();

    let text_ptrs = unsafe { std::slice::from_raw_parts(texts, text_count) };
    let text_strs: Vec<&str> = text_ptrs
        .iter()
        .map(|p| unsafe { CStr::from_ptr(*p) }.to_str().unwrap_or_default())
        .collect();

    match clip_score_inner(wrapper, path_str, &text_strs) {
        Ok(scores) => {
            let count = scores.len();
            let mut boxed = scores.into_boxed_slice();
            let ptr = boxed.as_mut_ptr();
            std::mem::forget(boxed);
            let result = ClipScoreResult { scores: ptr, count };
            Box::into_raw(Box::new(result))
        }
        Err(e) => {
            set_last_error(format!("CLIP scoring failed: {e}"));
            std::ptr::null_mut()
        }
    }
}

fn clip_score_inner(
    wrapper: &ClipPipelineWrapper,
    image_path: &str,
    texts: &[&str],
) -> anyhow::Result<Vec<f32>> {
    let img = image_utils::load_image(image_path, wrapper.image_size, &wrapper.device)?;
    let img = img.unsqueeze(0)?;

    // Tokenize each text and stack
    let mut all_ids = Vec::new();
    for text in texts {
        let ids = tokenize_clip(&wrapper.tokenizer, text, 77, &wrapper.device)?;
        all_ids.push(ids);
    }
    let input_ids = Tensor::cat(&all_ids, 0)?;

    let (logits_per_text, _logits_per_image) = wrapper.model.forward(&img, &input_ids)?;

    // logits_per_text is (num_texts, 1) — squeeze and softmax
    let logits = logits_per_text.squeeze(1)?;
    let probs = candle_nn::ops::softmax(&logits, 0)?;
    let scores: Vec<f32> = probs.to_vec1()?;

    Ok(scores)
}

#[no_mangle]
pub extern "C" fn run_clip_embed_image(
    wrapper: *mut ClipPipelineWrapper,
    image_path: *const c_char,
) -> *mut ClipEmbeddingResult {
    if wrapper.is_null() || image_path.is_null() {
        set_last_error("null pointer argument".to_string());
        return std::ptr::null_mut();
    }

    let wrapper = unsafe { &*wrapper };
    let path_str = unsafe { CStr::from_ptr(image_path) }
        .to_str()
        .unwrap_or_default();

    match clip_embed_image_inner(wrapper, path_str) {
        Ok(vec) => {
            let dim = vec.len();
            let mut boxed = vec.into_boxed_slice();
            let ptr = boxed.as_mut_ptr();
            std::mem::forget(boxed);
            let result = ClipEmbeddingResult { data: ptr, dim };
            Box::into_raw(Box::new(result))
        }
        Err(e) => {
            set_last_error(format!("CLIP image embedding failed: {e}"));
            std::ptr::null_mut()
        }
    }
}

fn clip_embed_image_inner(
    wrapper: &ClipPipelineWrapper,
    image_path: &str,
) -> anyhow::Result<Vec<f32>> {
    let img = image_utils::load_image(image_path, wrapper.image_size, &wrapper.device)?;
    let img = img.unsqueeze(0)?;
    let features = wrapper.model.get_image_features(&img)?;
    let vec: Vec<f32> = features.squeeze(0)?.to_vec1()?;
    Ok(vec)
}

#[no_mangle]
pub extern "C" fn run_clip_embed_text(
    wrapper: *mut ClipPipelineWrapper,
    text: *const c_char,
) -> *mut ClipEmbeddingResult {
    if wrapper.is_null() || text.is_null() {
        set_last_error("null pointer argument".to_string());
        return std::ptr::null_mut();
    }

    let wrapper = unsafe { &*wrapper };
    let text_str = unsafe { CStr::from_ptr(text) }
        .to_str()
        .unwrap_or_default();

    match clip_embed_text_inner(wrapper, text_str) {
        Ok(vec) => {
            let dim = vec.len();
            let mut boxed = vec.into_boxed_slice();
            let ptr = boxed.as_mut_ptr();
            std::mem::forget(boxed);
            let result = ClipEmbeddingResult { data: ptr, dim };
            Box::into_raw(Box::new(result))
        }
        Err(e) => {
            set_last_error(format!("CLIP text embedding failed: {e}"));
            std::ptr::null_mut()
        }
    }
}

fn clip_embed_text_inner(
    wrapper: &ClipPipelineWrapper,
    text: &str,
) -> anyhow::Result<Vec<f32>> {
    let input_ids = tokenize_clip(&wrapper.tokenizer, text, 77, &wrapper.device)?;
    let features = wrapper.model.get_text_features(&input_ids)?;
    let vec: Vec<f32> = features.squeeze(0)?.to_vec1()?;
    Ok(vec)
}

#[no_mangle]
pub extern "C" fn free_clip_pipeline(wrapper: *mut ClipPipelineWrapper) {
    if !wrapper.is_null() {
        unsafe {
            drop(Box::from_raw(wrapper));
        }
    }
}

#[no_mangle]
pub extern "C" fn free_clip_score_result(result: *mut ClipScoreResult) {
    if !result.is_null() {
        unsafe {
            let r = Box::from_raw(result);
            if !r.scores.is_null() {
                drop(Vec::from_raw_parts(r.scores, r.count, r.count));
            }
        }
    }
}

#[no_mangle]
pub extern "C" fn free_clip_embedding_result(result: *mut ClipEmbeddingResult) {
    if !result.is_null() {
        unsafe {
            let r = Box::from_raw(result);
            if !r.data.is_null() {
                drop(Vec::from_raw_parts(r.data, r.dim, r.dim));
            }
        }
    }
}
