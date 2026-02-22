use std::ffi::CStr;
use std::os::raw::c_char;

use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config as BertConfig, DTYPE};

use crate::{json_str, parse_config_json, set_last_error};

/// Opaque wrapper for an embedding pipeline.
pub struct EmbeddingPipelineWrapper {
    model: BertModel,
    tokenizer: tokenizers::Tokenizer,
    device: Device,
    normalize: bool,
}

/// Result of embedding.
#[repr(C)]
pub struct EmbeddingResult {
    pub data: *mut f32,
    pub dim: usize,
}

/// Result of batch embedding.
#[repr(C)]
pub struct BatchEmbeddingResult {
    pub data: *mut f32,
    pub dim: usize,
    pub count: usize,
}

fn l2_normalize(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

fn load_embedding_model(
    config: &serde_json::Value,
    device: &Device,
) -> anyhow::Result<(BertModel, tokenizers::Tokenizer, bool)> {
    let model_id = json_str(config, "model_id", "");
    if model_id.is_empty() {
        anyhow::bail!("model_id is required");
    }

    let normalize = config
        .get("normalize")
        .and_then(|v| v.as_bool())
        .unwrap_or(true);

    let cache_dir = config
        .get("cache_dir")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());

    let api = hf_hub::api::sync::Api::new()?;
    let repo = if let Some(ref dir) = cache_dir {
        let api = hf_hub::api::sync::ApiBuilder::new()
            .with_cache_dir(std::path::PathBuf::from(dir))
            .build()?;
        api.model(model_id.to_string())
    } else {
        api.model(model_id.to_string())
    };

    let config_path = repo.get("config.json")?;
    let tokenizer_path = repo.get("tokenizer.json")?;

    let bert_config: BertConfig = serde_json::from_reader(std::fs::File::open(&config_path)?)?;

    let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow::anyhow!("tokenizer load error: {e}"))?;

    // Load weights
    let weight_files = {
        if let Ok(p) = repo.get("model.safetensors") {
            vec![p]
        } else {
            let index_path = repo.get("model.safetensors.index.json")?;
            let index_data: serde_json::Value =
                serde_json::from_reader(std::fs::File::open(&index_path)?)?;
            let weight_map = index_data
                .get("weight_map")
                .and_then(|v| v.as_object())
                .ok_or_else(|| anyhow::anyhow!("no weight_map in index"))?;
            let mut files: Vec<String> = weight_map
                .values()
                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                .collect();
            files.sort();
            files.dedup();
            let mut paths = Vec::new();
            for f in &files {
                paths.push(repo.get(f)?);
            }
            paths
        }
    };

    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&weight_files, DTYPE, device)? };
    let model = BertModel::load(vb, &bert_config)?;

    Ok((model, tokenizer, normalize))
}

fn embed_text(wrapper: &EmbeddingPipelineWrapper, texts: &[&str]) -> anyhow::Result<Vec<Vec<f32>>> {
    let mut all_embeddings = Vec::with_capacity(texts.len());

    for text in texts {
        let encoding = wrapper
            .tokenizer
            .encode(*text, true)
            .map_err(|e| anyhow::anyhow!("tokenization error: {e}"))?;

        let token_ids = Tensor::new(encoding.get_ids(), &wrapper.device)?.unsqueeze(0)?;
        let token_type_ids = token_ids.zeros_like()?;

        let embeddings = wrapper.model.forward(&token_ids, &token_type_ids, None)?;

        // Mean pooling over sequence dimension
        let (_n_sentences, seq_len, _hidden) = embeddings.dims3()?;
        let sum = embeddings.sum(1)?;
        let mean = (sum / (seq_len as f64))?;

        let mut vec = mean.squeeze(0)?.to_vec1::<f32>()?;

        if wrapper.normalize {
            l2_normalize(&mut vec);
        }

        all_embeddings.push(vec);
    }

    Ok(all_embeddings)
}

/// Create a new embedding pipeline from JSON config.
///
/// Config fields:
/// - `model_id` (required): HF Hub model identifier (e.g. "sentence-transformers/all-MiniLM-L6-v2")
/// - `cache_dir` (optional): custom HF cache directory
/// - `normalize` (optional): L2-normalize embeddings, default true
#[no_mangle]
pub extern "C" fn new_embedding_pipeline(
    config_json: *const c_char,
) -> *mut EmbeddingPipelineWrapper {
    let config = match parse_config_json(config_json) {
        Ok(c) => c,
        Err(e) => {
            set_last_error(e);
            return std::ptr::null_mut();
        }
    };

    let device = Device::Cpu;

    match load_embedding_model(&config, &device) {
        Ok((model, tokenizer, normalize)) => {
            let wrapper = EmbeddingPipelineWrapper {
                model,
                tokenizer,
                device,
                normalize,
            };
            Box::into_raw(Box::new(wrapper))
        }
        Err(e) => {
            set_last_error(format!("failed to load embedding model: {e}"));
            std::ptr::null_mut()
        }
    }
}

/// Run embedding on a single text.
#[no_mangle]
pub extern "C" fn run_embedding(
    wrapper: *mut EmbeddingPipelineWrapper,
    text: *const c_char,
) -> *mut EmbeddingResult {
    if wrapper.is_null() || text.is_null() {
        set_last_error("null pointer argument".to_string());
        return std::ptr::null_mut();
    }

    let wrapper = unsafe { &*wrapper };
    let text_str = unsafe { CStr::from_ptr(text) }.to_str().unwrap_or_default();

    match embed_text(wrapper, &[text_str]) {
        Ok(embeddings) => {
            if let Some(emb) = embeddings.into_iter().next() {
                let dim = emb.len();
                let mut boxed = emb.into_boxed_slice();
                let ptr = boxed.as_mut_ptr();
                std::mem::forget(boxed);

                let result = EmbeddingResult { data: ptr, dim };
                Box::into_raw(Box::new(result))
            } else {
                set_last_error("no embedding produced".to_string());
                std::ptr::null_mut()
            }
        }
        Err(e) => {
            set_last_error(format!("embedding failed: {e}"));
            std::ptr::null_mut()
        }
    }
}

/// Run embedding on a batch of texts.
#[no_mangle]
pub extern "C" fn run_embedding_batch(
    wrapper: *mut EmbeddingPipelineWrapper,
    texts: *const *const c_char,
    count: usize,
) -> *mut BatchEmbeddingResult {
    if wrapper.is_null() || texts.is_null() || count == 0 {
        set_last_error("null pointer or zero count".to_string());
        return std::ptr::null_mut();
    }

    let wrapper = unsafe { &*wrapper };
    let text_ptrs = unsafe { std::slice::from_raw_parts(texts, count) };
    let text_strs: Vec<&str> = text_ptrs
        .iter()
        .map(|p| unsafe { CStr::from_ptr(*p) }.to_str().unwrap_or_default())
        .collect();

    match embed_text(wrapper, &text_strs) {
        Ok(embeddings) => {
            if embeddings.is_empty() {
                set_last_error("no embeddings produced".to_string());
                return std::ptr::null_mut();
            }
            let dim = embeddings[0].len();
            let count = embeddings.len();

            // Flatten into a single contiguous array
            let mut flat: Vec<f32> = Vec::with_capacity(dim * count);
            for emb in &embeddings {
                flat.extend_from_slice(emb);
            }
            let mut boxed = flat.into_boxed_slice();
            let ptr = boxed.as_mut_ptr();
            std::mem::forget(boxed);

            let result = BatchEmbeddingResult {
                data: ptr,
                dim,
                count,
            };
            Box::into_raw(Box::new(result))
        }
        Err(e) => {
            set_last_error(format!("batch embedding failed: {e}"));
            std::ptr::null_mut()
        }
    }
}

/// Free an embedding pipeline.
#[no_mangle]
pub extern "C" fn free_embedding_pipeline(wrapper: *mut EmbeddingPipelineWrapper) {
    if !wrapper.is_null() {
        unsafe {
            drop(Box::from_raw(wrapper));
        }
    }
}

/// Free an embedding result.
#[no_mangle]
pub extern "C" fn free_embedding_result(result: *mut EmbeddingResult) {
    if !result.is_null() {
        unsafe {
            let r = Box::from_raw(result);
            if !r.data.is_null() {
                drop(Vec::from_raw_parts(r.data, r.dim, r.dim));
            }
        }
    }
}

/// Free a batch embedding result.
#[no_mangle]
pub extern "C" fn free_batch_embedding_result(result: *mut BatchEmbeddingResult) {
    if !result.is_null() {
        unsafe {
            let r = Box::from_raw(result);
            if !r.data.is_null() {
                let total = r.dim * r.count;
                drop(Vec::from_raw_parts(r.data, total, total));
            }
        }
    }
}
