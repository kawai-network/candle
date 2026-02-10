use std::ffi::CString;
use std::os::raw::c_char;

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::marian;
use tokenizers::Tokenizer;

use crate::{create_hf_repo_with_revision, json_str, json_u64, parse_config_json, set_last_error};

/// Opaque wrapper for FFI.
pub struct TranslationPipelineInner {
    model: marian::MTModel,
    src_tokenizer: Tokenizer,
    tgt_tokenizer: Tokenizer,
    config: marian::Config,
    device: Device,
}

#[repr(C)]
pub struct TranslationPipelineWrapper {
    _opaque: *mut std::ffi::c_void,
}

#[repr(C)]
pub struct TranslationResult {
    pub text: *mut c_char,
}

/// Language pair configuration with all necessary info.
struct LangPairInfo {
    model_repo: &'static str,
    model_revision: Option<&'static str>,
    tokenizer_repo: &'static str,
    src_tokenizer_file: &'static str,
    tgt_tokenizer_file: &'static str,
    config_fn: fn() -> marian::Config,
}

fn get_lang_pair_info(pair: &str) -> Option<LangPairInfo> {
    match pair {
        "fr-en" => Some(LangPairInfo {
            model_repo: "Helsinki-NLP/opus-mt-fr-en",
            model_revision: Some("refs/pr/4"),
            tokenizer_repo: "lmz/candle-marian",
            src_tokenizer_file: "tokenizer-marian-base-fr.json",
            tgt_tokenizer_file: "tokenizer-marian-base-en.json",
            config_fn: marian::Config::opus_mt_fr_en,
        }),
        "en-fr" => Some(LangPairInfo {
            model_repo: "Helsinki-NLP/opus-mt-en-fr",
            model_revision: Some("refs/pr/9"),
            tokenizer_repo: "KeighBee/candle-marian",
            src_tokenizer_file: "tokenizer-marian-base-en-fr-en.json",
            tgt_tokenizer_file: "tokenizer-marian-base-en-fr-fr.json",
            config_fn: marian::Config::opus_mt_fr_en, // re-uses fr-en config shape
        }),
        "en-es" => Some(LangPairInfo {
            model_repo: "Helsinki-NLP/opus-mt-en-es",
            model_revision: Some("refs/pr/4"),
            tokenizer_repo: "KeighBee/candle-marian",
            src_tokenizer_file: "tokenizer-marian-base-en-es-en.json",
            tgt_tokenizer_file: "tokenizer-marian-base-en-es-es.json",
            config_fn: marian::Config::opus_mt_en_es,
        }),
        "en-zh" => Some(LangPairInfo {
            model_repo: "Helsinki-NLP/opus-mt-en-zh",
            model_revision: Some("refs/pr/13"),
            tokenizer_repo: "KeighBee/candle-marian",
            src_tokenizer_file: "tokenizer-marian-base-en-zh-en.json",
            tgt_tokenizer_file: "tokenizer-marian-base-en-zh-zh.json",
            config_fn: marian::Config::opus_mt_en_zh,
        }),
        "en-ru" => Some(LangPairInfo {
            model_repo: "Helsinki-NLP/opus-mt-en-ru",
            model_revision: Some("refs/pr/7"),
            tokenizer_repo: "KeighBee/candle-marian",
            src_tokenizer_file: "tokenizer-marian-base-en-ru-en.json",
            tgt_tokenizer_file: "tokenizer-marian-base-en-ru-ru.json",
            config_fn: marian::Config::opus_mt_en_ru,
        }),
        "en-hi" => Some(LangPairInfo {
            model_repo: "Helsinki-NLP/opus-mt-en-hi",
            model_revision: Some("refs/pr/3"),
            tokenizer_repo: "KeighBee/candle-marian",
            src_tokenizer_file: "tokenizer-marian-base-en-hi-en.json",
            tgt_tokenizer_file: "tokenizer-marian-base-en-hi-hi.json",
            config_fn: marian::Config::opus_mt_en_hi,
        }),
        _ => None,
    }
}

fn create_pipeline(config_json: *const c_char) -> anyhow::Result<Box<TranslationPipelineInner>> {
    let cfg = parse_config_json(config_json).map_err(|e| anyhow::anyhow!(e))?;

    let lang_pair = json_str(&cfg, "language_pair", "fr-en");
    let cache_dir = cfg.get("cache_dir").and_then(|v| v.as_str());

    let info = get_lang_pair_info(lang_pair).ok_or_else(|| {
        anyhow::anyhow!(
            "unsupported language pair: {lang_pair}. Supported: fr-en, en-fr, en-es, en-zh, en-ru, en-hi"
        )
    })?;

    let device = Device::Cpu;
    let config = (info.config_fn)();

    // Download model weights
    let model_repo = create_hf_repo_with_revision(info.model_repo, cache_dir, info.model_revision)?;
    let model_path = model_repo.get("model.safetensors")?;

    // Download tokenizers from the tokenizer repo
    let tok_repo = create_hf_repo_with_revision(info.tokenizer_repo, cache_dir, None)?;
    let src_tok_path = tok_repo.get(info.src_tokenizer_file)?;
    let tgt_tok_path = tok_repo.get(info.tgt_tokenizer_file)?;

    let src_tokenizer = Tokenizer::from_file(&src_tok_path)
        .map_err(|e| anyhow::anyhow!("src tokenizer load failed: {e}"))?;
    let tgt_tokenizer = Tokenizer::from_file(&tgt_tok_path)
        .map_err(|e| anyhow::anyhow!("tgt tokenizer load failed: {e}"))?;

    // Load model
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[model_path], DType::F32, &device)? };
    let model = marian::MTModel::new(&config, vb)?;

    Ok(Box::new(TranslationPipelineInner {
        model,
        src_tokenizer,
        tgt_tokenizer,
        config,
        device,
    }))
}

fn run_translate(inner: &mut TranslationPipelineInner, text: &str, params_json: Option<&str>) -> anyhow::Result<String> {
    let max_tokens = if let Some(pj) = params_json {
        let params: serde_json::Value = serde_json::from_str(pj)?;
        json_u64(&params, "max_tokens", 512) as usize
    } else {
        512
    };

    // Tokenize source text (add EOS at end)
    let mut tokens = inner
        .src_tokenizer
        .encode(text, true)
        .map_err(|e| anyhow::anyhow!("tokenize failed: {e}"))?
        .get_ids()
        .to_vec();
    tokens.push(inner.config.eos_token_id);

    let input_tensor = Tensor::new(tokens.as_slice(), &inner.device)?.unsqueeze(0)?;

    // Encode
    let encoder_xs = inner.model.encoder().forward(&input_tensor, 0)?;

    // Decode
    let mut token_ids: Vec<u32> = vec![inner.config.decoder_start_token_id];
    let mut logits_processor = LogitsProcessor::new(1337, None, None);

    for index in 0..max_tokens {
        let context_size = if index >= 1 { 1 } else { token_ids.len() };
        let start_pos = token_ids.len().saturating_sub(context_size);
        let input_ids = Tensor::new(&token_ids[start_pos..], &inner.device)?.unsqueeze(0)?;

        let logits = inner.model.decode(&input_ids, &encoder_xs, start_pos)?;
        let logits = logits.squeeze(0)?;
        let logits = logits.get(logits.dim(0)? - 1)?;

        let token = logits_processor.sample(&logits)?;
        if token == inner.config.eos_token_id || token == inner.config.forced_eos_token_id {
            break;
        }
        token_ids.push(token);
    }

    // Decode output tokens (skip decoder_start token)
    let output_ids = &token_ids[1..];
    let text = inner
        .tgt_tokenizer
        .decode(output_ids, true)
        .map_err(|e| anyhow::anyhow!("decode failed: {e}"))?;

    // Reset KV cache for next call
    inner.model.reset_kv_cache();

    Ok(text)
}

// --- FFI ---

#[no_mangle]
pub extern "C" fn new_translation_pipeline(config_json: *const c_char) -> *mut TranslationPipelineWrapper {
    match create_pipeline(config_json) {
        Ok(inner) => {
            let wrapper = Box::new(TranslationPipelineWrapper {
                _opaque: Box::into_raw(inner) as *mut std::ffi::c_void,
            });
            Box::into_raw(wrapper)
        }
        Err(e) => {
            set_last_error(format!("Translation pipeline creation failed: {e}"));
            std::ptr::null_mut()
        }
    }
}

#[no_mangle]
pub extern "C" fn run_translation(
    wrapper: *mut TranslationPipelineWrapper,
    text: *const c_char,
    params_json: *const c_char,
) -> *mut TranslationResult {
    if wrapper.is_null() || text.is_null() {
        set_last_error("null pointer passed to run_translation".to_string());
        return std::ptr::null_mut();
    }

    let inner = unsafe { &mut *((*wrapper)._opaque as *mut TranslationPipelineInner) };
    let text_str = unsafe { std::ffi::CStr::from_ptr(text) }
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

    match run_translate(inner, text_str, params) {
        Ok(result_text) => {
            let c_text = CString::new(result_text).unwrap_or_default();
            let result = Box::new(TranslationResult {
                text: c_text.into_raw(),
            });
            Box::into_raw(result)
        }
        Err(e) => {
            set_last_error(format!("Translation failed: {e}"));
            std::ptr::null_mut()
        }
    }
}

#[no_mangle]
pub extern "C" fn free_translation_pipeline(wrapper: *mut TranslationPipelineWrapper) {
    if !wrapper.is_null() {
        unsafe {
            let w = Box::from_raw(wrapper);
            if !w._opaque.is_null() {
                drop(Box::from_raw(w._opaque as *mut TranslationPipelineInner));
            }
        }
    }
}

#[no_mangle]
pub extern "C" fn free_translation_result(result: *mut TranslationResult) {
    if !result.is_null() {
        unsafe {
            let r = Box::from_raw(result);
            if !r.text.is_null() {
                drop(CString::from_raw(r.text));
            }
        }
    }
}
