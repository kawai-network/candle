use std::ffi::CString;
use std::os::raw::c_char;

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::t5;
use tokenizers::Tokenizer;

use crate::{
    create_hf_repo_with_revision, json_f64, json_str, json_u64, load_weight_files,
    parse_config_json, set_last_error,
};

/// Opaque wrapper for FFI.
pub struct T5PipelineInner {
    model: t5::T5ForConditionalGeneration,
    tokenizer: Tokenizer,
    config: t5::Config,
    device: Device,
}

#[repr(C)]
pub struct T5PipelineWrapper {
    _opaque: *mut std::ffi::c_void,
}

#[repr(C)]
pub struct T5Result {
    pub text: *mut c_char,
}

/// Known T5 model variants with their default revisions.
fn default_revision(model_id: &str) -> &'static str {
    match model_id {
        "t5-small" => "refs/pr/15",
        "t5-base" | "t5-large" | "t5-3b" | "t5-11b" => "main",
        "google/flan-t5-small"
        | "google/flan-t5-base"
        | "google/flan-t5-large"
        | "google/flan-t5-xl"
        | "google/flan-t5-xxl" => "main",
        "google/flan-ul2" => "main",
        _ if model_id.starts_with("google/mt5-") => {
            // mt5 models typically use a specific PR branch
            "main"
        }
        _ => "main",
    }
}

/// Determine if the model needs sharded weight files.
fn needs_sharded_weights(model_id: &str) -> bool {
    matches!(
        model_id,
        "google/flan-t5-xxl" | "google/flan-ul2" | "t5-3b" | "t5-11b"
    )
}

fn create_pipeline(config_json: *const c_char) -> anyhow::Result<Box<T5PipelineInner>> {
    let cfg = parse_config_json(config_json).map_err(|e| anyhow::anyhow!(e))?;

    let model_id = json_str(&cfg, "model_id", "t5-small");
    let cache_dir = cfg.get("cache_dir").and_then(|v| v.as_str());
    let revision = cfg.get("revision").and_then(|v| v.as_str());
    let disable_cache = cfg
        .get("disable_cache")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    let device = Device::Cpu;

    // Determine revision
    let effective_revision = revision.unwrap_or_else(|| default_revision(model_id));

    // Download model files
    let repo = create_hf_repo_with_revision(model_id, cache_dir, Some(effective_revision))?;

    let config_path = repo.get("config.json")?;
    let tokenizer_path = repo.get("tokenizer.json")?;

    let weight_files = if needs_sharded_weights(model_id) {
        load_weight_files(&repo)?
    } else {
        vec![repo.get("model.safetensors")?]
    };

    // Parse config
    let config_str = std::fs::read_to_string(&config_path)?;
    let mut config: t5::Config = serde_json::from_str(&config_str)?;
    config.use_cache = !disable_cache;

    // Load tokenizer
    let tokenizer = Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow::anyhow!("tokenizer load failed: {e}"))?;

    // Load model
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&weight_files, DType::F32, &device)? };
    let model = t5::T5ForConditionalGeneration::load(vb, &config)?;

    Ok(Box::new(T5PipelineInner {
        model,
        tokenizer,
        config,
        device,
    }))
}

fn run_generate(
    inner: &mut T5PipelineInner,
    prompt: &str,
    params_json: Option<&str>,
) -> anyhow::Result<String> {
    let max_tokens = if let Some(pj) = params_json {
        let params: serde_json::Value = serde_json::from_str(pj)?;
        let max = json_u64(&params, "max_tokens", 256) as usize;
        let _temperature = json_f64(&params, "temperature", 0.0);
        let _top_p = params.get("top_p").and_then(|v| v.as_f64());
        let _repeat_penalty = json_f64(&params, "repeat_penalty", 1.1) as f32;
        max
    } else {
        256
    };

    let temperature = if let Some(pj) = params_json {
        let params: serde_json::Value = serde_json::from_str(pj)?;
        let t = json_f64(&params, "temperature", 0.0);
        if t <= 0.0 {
            None
        } else {
            Some(t)
        }
    } else {
        None
    };

    let top_p = if let Some(pj) = params_json {
        let params: serde_json::Value = serde_json::from_str(pj)?;
        params.get("top_p").and_then(|v| v.as_f64())
    } else {
        None
    };

    let repeat_penalty = if let Some(pj) = params_json {
        let params: serde_json::Value = serde_json::from_str(pj)?;
        json_f64(&params, "repeat_penalty", 1.1) as f32
    } else {
        1.1
    };

    // Tokenize input
    let tokens = inner
        .tokenizer
        .encode(prompt, true)
        .map_err(|e| anyhow::anyhow!("tokenize failed: {e}"))?
        .get_ids()
        .to_vec();
    let input_token_ids = Tensor::new(&tokens[..], &inner.device)?.unsqueeze(0)?;

    // Encode
    let encoder_output = inner.model.encode(&input_token_ids)?;

    // Initialize decoder with start token
    let decoder_start = inner
        .config
        .decoder_start_token_id
        .unwrap_or(inner.config.pad_token_id) as u32;
    let mut output_token_ids: Vec<u32> = vec![decoder_start];

    let mut logits_processor = LogitsProcessor::new(299792458, temperature, top_p);

    for index in 0..max_tokens {
        let decoder_token_ids = if index == 0 || !inner.config.use_cache {
            Tensor::new(output_token_ids.as_slice(), &inner.device)?.unsqueeze(0)?
        } else {
            let last_token = *output_token_ids.last().unwrap();
            Tensor::new(&[last_token], &inner.device)?.unsqueeze(0)?
        };

        let logits = inner
            .model
            .decode(&decoder_token_ids, &encoder_output)?
            .squeeze(0)?;

        // Apply repeat penalty
        let logits = if repeat_penalty != 1.0 {
            let start_at = output_token_ids.len().saturating_sub(64);
            candle_transformers::utils::apply_repeat_penalty(
                &logits,
                repeat_penalty,
                &output_token_ids[start_at..],
            )?
        } else {
            logits
        };

        let next_token_id = logits_processor.sample(&logits)?;
        if next_token_id as usize == inner.config.eos_token_id {
            break;
        }
        output_token_ids.push(next_token_id);
    }

    // Decode output tokens (skip decoder_start token)
    let output_ids = &output_token_ids[1..];
    let text = inner
        .tokenizer
        .decode(output_ids, true)
        .map_err(|e| anyhow::anyhow!("decode failed: {e}"))?;

    // Clear KV cache for next call
    inner.model.clear_kv_cache();

    Ok(text)
}

// --- FFI ---

#[no_mangle]
pub extern "C" fn new_t5_pipeline(config_json: *const c_char) -> *mut T5PipelineWrapper {
    match create_pipeline(config_json) {
        Ok(inner) => {
            let wrapper = Box::new(T5PipelineWrapper {
                _opaque: Box::into_raw(inner) as *mut std::ffi::c_void,
            });
            Box::into_raw(wrapper)
        }
        Err(e) => {
            set_last_error(format!("T5 pipeline creation failed: {e}"));
            std::ptr::null_mut()
        }
    }
}

#[no_mangle]
pub extern "C" fn run_t5_generate(
    wrapper: *mut T5PipelineWrapper,
    prompt: *const c_char,
    params_json: *const c_char,
) -> *mut T5Result {
    if wrapper.is_null() || prompt.is_null() {
        set_last_error("null pointer passed to run_t5_generate".to_string());
        return std::ptr::null_mut();
    }

    let inner = unsafe { &mut *((*wrapper)._opaque as *mut T5PipelineInner) };
    let prompt_str = unsafe { std::ffi::CStr::from_ptr(prompt) }
        .to_str()
        .unwrap_or("");

    let params = if params_json.is_null() {
        None
    } else {
        Some(
            unsafe { std::ffi::CStr::from_ptr(params_json) }
                .to_str()
                .unwrap_or(""),
        )
    };

    match run_generate(inner, prompt_str, params) {
        Ok(text) => {
            let c_text = CString::new(text).unwrap_or_default();
            let result = Box::new(T5Result {
                text: c_text.into_raw(),
            });
            Box::into_raw(result)
        }
        Err(e) => {
            set_last_error(format!("T5 generation failed: {e}"));
            std::ptr::null_mut()
        }
    }
}

#[no_mangle]
pub extern "C" fn free_t5_pipeline(wrapper: *mut T5PipelineWrapper) {
    if !wrapper.is_null() {
        unsafe {
            let w = Box::from_raw(wrapper);
            if !w._opaque.is_null() {
                drop(Box::from_raw(w._opaque as *mut T5PipelineInner));
            }
        }
    }
}

#[no_mangle]
pub extern "C" fn free_t5_result(result: *mut T5Result) {
    if !result.is_null() {
        unsafe {
            let r = Box::from_raw(result);
            if !r.text.is_null() {
                drop(CString::from_raw(r.text));
            }
        }
    }
}
