use std::ffi::{CStr, CString};
use std::os::raw::c_char;

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::LogitsProcessor;

use crate::{json_f64, json_str, json_u64, parse_config_json, set_last_error};

/// Opaque wrapper for a text generation pipeline.
pub struct TextGenerationPipelineWrapper {
    model: Box<dyn TextGenModel>,
    tokenizer: tokenizers::Tokenizer,
    device: Device,
}

/// Trait abstracting over different text generation model architectures.
trait TextGenModel: Send {
    fn forward(&mut self, input_ids: &Tensor, pos: usize) -> anyhow::Result<Tensor>;
    /// Reset the KV cache for a new generation. Not all models need this.
    fn reset_cache(&mut self) {}
}

// --- Phi3 model implementation ---
struct PhiModel {
    model: candle_transformers::models::phi3::Model,
}

impl TextGenModel for PhiModel {
    fn forward(&mut self, input_ids: &Tensor, pos: usize) -> anyhow::Result<Tensor> {
        Ok(self.model.forward(input_ids, pos)?)
    }
    fn reset_cache(&mut self) {
        self.model.clear_kv_cache();
    }
}

// --- Llama model implementation ---
struct LlamaModel {
    model: candle_transformers::models::llama::Llama,
    cache: candle_transformers::models::llama::Cache,
    // Store params to recreate cache
    dtype: DType,
    config: candle_transformers::models::llama::Config,
    device: Device,
}

impl TextGenModel for LlamaModel {
    fn forward(&mut self, input_ids: &Tensor, pos: usize) -> anyhow::Result<Tensor> {
        Ok(self.model.forward(input_ids, pos, &mut self.cache)?)
    }
    fn reset_cache(&mut self) {
        // Recreate the cache to clear it
        if let Ok(cache) =
            candle_transformers::models::llama::Cache::new(true, self.dtype, &self.config, &self.device)
        {
            self.cache = cache;
        }
    }
}

// --- Mistral model implementation ---
struct MistralModel {
    model: candle_transformers::models::mistral::Model,
}

impl TextGenModel for MistralModel {
    fn forward(&mut self, input_ids: &Tensor, pos: usize) -> anyhow::Result<Tensor> {
        Ok(self.model.forward(input_ids, pos)?)
    }
    fn reset_cache(&mut self) {
        self.model.clear_kv_cache();
    }
}

// --- Gemma model implementation ---
struct GemmaModel {
    model: candle_transformers::models::gemma::Model,
}

impl TextGenModel for GemmaModel {
    fn forward(&mut self, input_ids: &Tensor, pos: usize) -> anyhow::Result<Tensor> {
        Ok(self.model.forward(input_ids, pos)?)
    }
    fn reset_cache(&mut self) {
        self.model.clear_kv_cache();
    }
}

// --- Qwen2 model implementation ---
struct Qwen2Model {
    model: candle_transformers::models::qwen2::ModelForCausalLM,
}

impl TextGenModel for Qwen2Model {
    fn forward(&mut self, input_ids: &Tensor, pos: usize) -> anyhow::Result<Tensor> {
        Ok(self.model.forward(input_ids, pos)?)
    }
    fn reset_cache(&mut self) {
        self.model.clear_kv_cache();
    }
}

// --- Falcon model implementation ---
struct FalconModel {
    model: candle_transformers::models::falcon::Falcon,
}

impl TextGenModel for FalconModel {
    fn forward(&mut self, input_ids: &Tensor, _pos: usize) -> anyhow::Result<Tensor> {
        Ok(self.model.forward(input_ids)?)
    }
    fn reset_cache(&mut self) {
        self.model.clear_kv_cache();
    }
}

// --- StableLM model implementation ---
// StableLM uses internal KV cache in attention layers;
// cache is managed internally and doesn't need external clearing.
struct StableLmModel {
    model: candle_transformers::models::stable_lm::Model,
}

impl TextGenModel for StableLmModel {
    fn forward(&mut self, input_ids: &Tensor, pos: usize) -> anyhow::Result<Tensor> {
        Ok(self.model.forward(input_ids, pos)?)
    }
    // No clear_kv_cache available for StableLM; it manages cache internally
}

/// Result of text generation.
#[repr(C)]
pub struct TextGenerationResult {
    pub text: *mut c_char,
    pub tokens_generated: usize,
}

fn load_model(
    config: &serde_json::Value,
    device: &Device,
) -> anyhow::Result<(Box<dyn TextGenModel>, tokenizers::Tokenizer)> {
    let model_id = json_str(config, "model_id", "");
    if model_id.is_empty() {
        anyhow::bail!("model_id is required");
    }

    let cache_dir = config
        .get("cache_dir")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());

    // Download model files from HF Hub
    let api = hf_hub::api::sync::Api::new()?;
    let repo = if let Some(ref dir) = cache_dir {
        let api = hf_hub::api::sync::ApiBuilder::new()
            .with_cache_dir(std::path::PathBuf::from(dir))
            .build()?;
        api.model(model_id.to_string())
    } else {
        api.model(model_id.to_string())
    };

    // Load config.json to detect architecture
    let config_path = repo.get("config.json")?;
    let config_data: serde_json::Value =
        serde_json::from_reader(std::fs::File::open(&config_path)?)?;

    let architectures = config_data
        .get("architectures")
        .and_then(|v| v.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|v| v.as_str())
                .map(|s| s.to_lowercase())
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();

    let model_type = config_data
        .get("model_type")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_lowercase();

    // Load tokenizer
    let tokenizer_path = repo.get("tokenizer.json")?;
    let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow::anyhow!("failed to load tokenizer: {e}"))?;

    // Load model weights (safetensors)
    let weight_files = {
        // Try single file first
        if let Ok(p) = repo.get("model.safetensors") {
            vec![p]
        } else {
            // Try sharded - look for index
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

    let dtype = DType::F32;
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&weight_files, dtype, device)? };

    // Match architecture and create the appropriate model
    let arch_str = architectures.first().map(|s| s.as_str()).unwrap_or("");

    let model: Box<dyn TextGenModel> =
        if arch_str.contains("phi3") || model_type == "phi3" || model_type == "phi" {
            let cfg: candle_transformers::models::phi3::Config =
                serde_json::from_reader(std::fs::File::open(&config_path)?)?;
            let m = candle_transformers::models::phi3::Model::new(&cfg, vb)?;
            Box::new(PhiModel { model: m })
        } else if arch_str.contains("llama") || model_type == "llama" {
            let llama_cfg: candle_transformers::models::llama::LlamaConfig =
                serde_json::from_reader(std::fs::File::open(&config_path)?)?;
            let cfg = llama_cfg.into_config(false);
            let cache = candle_transformers::models::llama::Cache::new(true, dtype, &cfg, device)?;
            let m = candle_transformers::models::llama::Llama::load(vb, &cfg)?;
            Box::new(LlamaModel {
                model: m,
                cache,
                dtype,
                config: cfg,
                device: device.clone(),
            })
        } else if arch_str.contains("mistral") || model_type == "mistral" {
            let cfg: candle_transformers::models::mistral::Config =
                serde_json::from_reader(std::fs::File::open(&config_path)?)?;
            let m = candle_transformers::models::mistral::Model::new(&cfg, vb)?;
            Box::new(MistralModel { model: m })
        } else if arch_str.contains("gemma") || model_type == "gemma" {
            let cfg: candle_transformers::models::gemma::Config =
                serde_json::from_reader(std::fs::File::open(&config_path)?)?;
            let m = candle_transformers::models::gemma::Model::new(false, &cfg, vb)?;
            Box::new(GemmaModel { model: m })
        } else if arch_str.contains("qwen2") || model_type == "qwen2" {
            let cfg: candle_transformers::models::qwen2::Config =
                serde_json::from_reader(std::fs::File::open(&config_path)?)?;
            let m = candle_transformers::models::qwen2::ModelForCausalLM::new(&cfg, vb)?;
            Box::new(Qwen2Model { model: m })
        } else if arch_str.contains("falcon") || model_type == "falcon" {
            let cfg: candle_transformers::models::falcon::Config =
                serde_json::from_reader(std::fs::File::open(&config_path)?)?;
            let m = candle_transformers::models::falcon::Falcon::load(vb, cfg)?;
            Box::new(FalconModel { model: m })
        } else if arch_str.contains("stablelm") || model_type == "stablelm" {
            let cfg: candle_transformers::models::stable_lm::Config =
                serde_json::from_reader(std::fs::File::open(&config_path)?)?;
            let m = candle_transformers::models::stable_lm::Model::new(&cfg, vb)?;
            Box::new(StableLmModel { model: m })
        } else {
            anyhow::bail!(
                "unsupported model architecture: {:?} (model_type={})",
                architectures,
                model_type
            );
        };

    Ok((model, tokenizer))
}

fn generate_text(
    wrapper: &mut TextGenerationPipelineWrapper,
    prompt: &str,
    max_tokens: usize,
    temperature: f64,
    top_p: f64,
    repeat_penalty: f32,
    seed: u64,
) -> anyhow::Result<(String, usize)> {
    let encoding = wrapper
        .tokenizer
        .encode(prompt, true)
        .map_err(|e| anyhow::anyhow!("tokenization error: {e}"))?;
    let mut tokens = encoding.get_ids().to_vec();

    let mut logits_processor = LogitsProcessor::new(seed, Some(temperature), Some(top_p));

    wrapper.model.reset_cache();

    // Process prompt tokens
    let input = Tensor::new(&tokens[..], &wrapper.device)?.unsqueeze(0)?;
    let logits = wrapper.model.forward(&input, 0)?;
    let logits = logits.squeeze(0)?.to_dtype(DType::F32)?;

    let mut next_token = logits_processor.sample(&logits)?;
    tokens.push(next_token);

    let eos_token_id = wrapper
        .tokenizer
        .token_to_id("</s>")
        .or_else(|| wrapper.tokenizer.token_to_id("<|endoftext|>"))
        .or_else(|| wrapper.tokenizer.token_to_id("<|end|>"));

    let mut generated_count = 1usize;

    for _i in 0..max_tokens.saturating_sub(1) {
        if Some(next_token) == eos_token_id {
            break;
        }

        let input = Tensor::new(&[next_token], &wrapper.device)?.unsqueeze(0)?;
        let logits = wrapper
            .model
            .forward(&input, tokens.len().saturating_sub(1))?;
        let logits = logits.squeeze(0)?.to_dtype(DType::F32)?;

        // Apply repeat penalty
        if repeat_penalty > 1.0 {
            let start = tokens.len().saturating_sub(64);
            candle_transformers::utils::apply_repeat_penalty(
                &logits,
                repeat_penalty,
                &tokens[start..],
            )?;
        }

        next_token = logits_processor.sample(&logits)?;
        tokens.push(next_token);
        generated_count += 1;
    }

    // Decode generated tokens (skip original prompt tokens)
    let prompt_len = encoding.get_ids().len();
    let generated_tokens = &tokens[prompt_len..];
    let text = wrapper
        .tokenizer
        .decode(generated_tokens, true)
        .map_err(|e| anyhow::anyhow!("decode error: {e}"))?;

    Ok((text, generated_count))
}

/// Create a new text generation pipeline from JSON config.
#[no_mangle]
pub extern "C" fn new_text_generation_pipeline(
    config_json: *const c_char,
) -> *mut TextGenerationPipelineWrapper {
    let config = match parse_config_json(config_json) {
        Ok(c) => c,
        Err(e) => {
            set_last_error(e);
            return std::ptr::null_mut();
        }
    };

    let device = Device::Cpu;

    match load_model(&config, &device) {
        Ok((model, tokenizer)) => {
            let wrapper = TextGenerationPipelineWrapper {
                model,
                tokenizer,
                device,
            };
            Box::into_raw(Box::new(wrapper))
        }
        Err(e) => {
            set_last_error(format!("failed to load model: {e}"));
            std::ptr::null_mut()
        }
    }
}

/// Run text generation.
#[no_mangle]
pub extern "C" fn run_text_generation(
    wrapper: *mut TextGenerationPipelineWrapper,
    prompt: *const c_char,
    params_json: *const c_char,
) -> *mut TextGenerationResult {
    if wrapper.is_null() || prompt.is_null() {
        set_last_error("null pointer argument".to_string());
        return std::ptr::null_mut();
    }

    let wrapper = unsafe { &mut *wrapper };
    let prompt_str = unsafe { CStr::from_ptr(prompt) }
        .to_str()
        .unwrap_or_default();

    let params = if params_json.is_null() {
        serde_json::Value::Object(serde_json::Map::new())
    } else {
        match parse_config_json(params_json) {
            Ok(p) => p,
            Err(e) => {
                set_last_error(e);
                return std::ptr::null_mut();
            }
        }
    };

    let max_tokens = json_u64(&params, "max_tokens", 100) as usize;
    let temperature = json_f64(&params, "temperature", 0.8);
    let top_p = json_f64(&params, "top_p", 0.9);
    let repeat_penalty = json_f64(&params, "repeat_penalty", 1.1) as f32;
    let seed = json_u64(&params, "seed", 42);

    match generate_text(
        wrapper,
        prompt_str,
        max_tokens,
        temperature,
        top_p,
        repeat_penalty,
        seed,
    ) {
        Ok((text, count)) => {
            let c_text = CString::new(text).unwrap_or_default();
            let result = TextGenerationResult {
                text: c_text.into_raw(),
                tokens_generated: count,
            };
            Box::into_raw(Box::new(result))
        }
        Err(e) => {
            set_last_error(format!("generation failed: {e}"));
            std::ptr::null_mut()
        }
    }
}

/// Free a text generation pipeline.
#[no_mangle]
pub extern "C" fn free_text_generation_pipeline(wrapper: *mut TextGenerationPipelineWrapper) {
    if !wrapper.is_null() {
        unsafe {
            drop(Box::from_raw(wrapper));
        }
    }
}

/// Free a text generation result.
#[no_mangle]
pub extern "C" fn free_text_generation_result(result: *mut TextGenerationResult) {
    if !result.is_null() {
        unsafe {
            let r = Box::from_raw(result);
            if !r.text.is_null() {
                drop(CString::from_raw(r.text));
            }
        }
    }
}
