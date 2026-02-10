use std::ffi::{CStr, CString};
use std::os::raw::c_char;

use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::ops::softmax;
use candle_nn::VarBuilder;
use candle_transformers::models::whisper::{self as m, audio, Config};
use rand::{distributions::WeightedIndex, prelude::Distribution, SeedableRng};

use crate::{json_f64, json_str, json_u64, parse_config_json, set_last_error, create_hf_repo, load_weight_files};

pub struct WhisperPipelineWrapper {
    model: m::model::Whisper,
    config: Config,
    tokenizer: tokenizers::Tokenizer,
    mel_filters: Vec<f32>,
    device: Device,
}

/// Whisper transcription result.
#[repr(C)]
pub struct WhisperResult {
    pub text: *mut c_char,
    pub segments: *mut WhisperSegment,
    pub segment_count: usize,
}

/// A single timestamped segment.
#[repr(C)]
pub struct WhisperSegment {
    pub text: *mut c_char,
    pub start: f64,
    pub end: f64,
}

fn load_whisper_model(
    config: &serde_json::Value,
    device: &Device,
) -> anyhow::Result<WhisperPipelineWrapper> {
    let model_id = json_str(config, "model_id", "openai/whisper-tiny");
    let cache_dir = config.get("cache_dir").and_then(|v| v.as_str());

    let repo = create_hf_repo(model_id, cache_dir)?;

    let config_path = repo.get("config.json")?;
    let whisper_config: Config =
        serde_json::from_reader(std::fs::File::open(&config_path)?)?;

    let tokenizer_path = repo.get("tokenizer.json")?;
    let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow::anyhow!("failed to load tokenizer: {e}"))?;

    let weight_files = load_weight_files(&repo)?;
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&weight_files, DType::F32, device)? };

    let model = m::model::Whisper::load(&vb, whisper_config.clone())?;

    // Load mel filters
    let mel_bytes: &[u8] = match whisper_config.num_mel_bins {
        80 => include_bytes!("melfilters.bytes"),
        128 => include_bytes!("melfilters128.bytes"),
        n => anyhow::bail!("unsupported num_mel_bins: {}", n),
    };
    let mut mel_filters = vec![0f32; mel_bytes.len() / 4];
    byteorder::LittleEndian::read_f32_into(mel_bytes, &mut mel_filters);

    Ok(WhisperPipelineWrapper {
        model,
        config: whisper_config,
        tokenizer,
        mel_filters,
        device: device.clone(),
    })
}

use byteorder::ByteOrder;

fn transcribe(
    wrapper: &mut WhisperPipelineWrapper,
    audio_path: &str,
    temperature: f64,
    language: &str,
) -> anyhow::Result<(String, Vec<(String, f64, f64)>)> {
    // Load WAV file
    let reader = hound::WavReader::open(audio_path)?;
    let spec = reader.spec();
    let sample_rate = spec.sample_rate;

    let pcm_data: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => {
            reader.into_samples::<f32>().filter_map(|s| s.ok()).collect()
        }
        hound::SampleFormat::Int => {
            let bits = spec.bits_per_sample;
            let max = (1 << (bits - 1)) as f32;
            reader
                .into_samples::<i32>()
                .filter_map(|s| s.ok())
                .map(|s| s as f32 / max)
                .collect()
        }
    };

    if sample_rate != m::SAMPLE_RATE as u32 {
        anyhow::bail!(
            "audio must be {}Hz, got {}Hz",
            m::SAMPLE_RATE,
            sample_rate
        );
    }

    // Convert to mel spectrogram
    let mel = audio::pcm_to_mel(&wrapper.config, &pcm_data, &wrapper.mel_filters);
    let mel_len = mel.len();
    let mel = Tensor::from_vec(
        mel,
        (1, wrapper.config.num_mel_bins, mel_len / wrapper.config.num_mel_bins),
        &wrapper.device,
    )?;

    // Get special token IDs
    let sot_token = token_id(&wrapper.tokenizer, m::SOT_TOKEN)?;
    let eot_token = token_id(&wrapper.tokenizer, m::EOT_TOKEN)?;
    let transcribe_token = token_id(&wrapper.tokenizer, m::TRANSCRIBE_TOKEN)?;
    let no_timestamps_token = token_id(&wrapper.tokenizer, m::NO_TIMESTAMPS_TOKEN)?;

    let language_token = if !language.is_empty() {
        let lang_str = format!("<|{}|>", language);
        wrapper.tokenizer.token_to_id(&lang_str)
    } else {
        wrapper.tokenizer.token_to_id("<|en|>")
    };

    // Build initial token sequence
    let mut tokens: Vec<u32> = vec![sot_token];
    if let Some(lt) = language_token {
        tokens.push(lt);
    }
    tokens.push(transcribe_token);
    tokens.push(no_timestamps_token);

    // Encode audio
    wrapper.model.reset_kv_cache();
    let audio_features = wrapper.model.encoder.forward(&mel, true)?;

    // Build suppress tokens mask
    let suppress_tokens: Vec<f32> = (0..wrapper.config.vocab_size)
        .map(|i| {
            if wrapper.config.suppress_tokens.contains(&(i as u32)) {
                f32::NEG_INFINITY
            } else {
                0f32
            }
        })
        .collect();
    let suppress_tokens = Tensor::new(suppress_tokens, &wrapper.device)?;

    let sample_len = wrapper.config.max_target_positions / 2;
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);

    // Autoregressive decoding
    for i in 0..sample_len {
        let tokens_t = Tensor::new(tokens.as_slice(), &wrapper.device)?.unsqueeze(0)?;
        let ys = wrapper.model.decoder.forward(&tokens_t, &audio_features, i == 0)?;
        let logits = wrapper.model.decoder.final_linear(&ys.i(..1)?)?.i(0)?.i(ys.dim(1)? - 1)?;

        let logits = logits.broadcast_add(&suppress_tokens)?;

        let next_token = if temperature > 0.0 {
            let prs = softmax(&(&logits / temperature)?, 0)?;
            let prs_vec: Vec<f32> = prs.to_vec1()?;
            let distr = WeightedIndex::new(&prs_vec)
                .map_err(|e| anyhow::anyhow!("sampling error: {e}"))?;
            distr.sample(&mut rng) as u32
        } else {
            let logits_vec: Vec<f32> = logits.to_vec1()?;
            logits_vec
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i as u32)
                .unwrap_or(eot_token)
        };

        tokens.push(next_token);

        if next_token == eot_token || tokens.len() > wrapper.config.max_target_positions {
            break;
        }
    }

    // Decode tokens to text (skip prompt tokens)
    let prompt_len = if language_token.is_some() { 4 } else { 3 };
    let generated = &tokens[prompt_len..];
    // Filter out eot
    let generated: Vec<u32> = generated
        .iter()
        .copied()
        .filter(|&t| t != eot_token)
        .collect();

    let text = wrapper
        .tokenizer
        .decode(&generated, true)
        .map_err(|e| anyhow::anyhow!("decode error: {e}"))?;

    // For now, return as a single segment
    let duration = pcm_data.len() as f64 / m::SAMPLE_RATE as f64;
    let segments = vec![(text.clone(), 0.0, duration)];

    Ok((text, segments))
}

fn token_id(tokenizer: &tokenizers::Tokenizer, token: &str) -> anyhow::Result<u32> {
    tokenizer
        .token_to_id(token)
        .ok_or_else(|| anyhow::anyhow!("token not found: {}", token))
}

#[no_mangle]
pub extern "C" fn new_whisper_pipeline(
    config_json: *const c_char,
) -> *mut WhisperPipelineWrapper {
    let config = match parse_config_json(config_json) {
        Ok(c) => c,
        Err(e) => {
            set_last_error(e);
            return std::ptr::null_mut();
        }
    };

    let device = Device::Cpu;

    match load_whisper_model(&config, &device) {
        Ok(wrapper) => Box::into_raw(Box::new(wrapper)),
        Err(e) => {
            set_last_error(format!("failed to load whisper model: {e}"));
            std::ptr::null_mut()
        }
    }
}

#[no_mangle]
pub extern "C" fn run_whisper_transcribe(
    wrapper: *mut WhisperPipelineWrapper,
    audio_path: *const c_char,
    params_json: *const c_char,
) -> *mut WhisperResult {
    if wrapper.is_null() || audio_path.is_null() {
        set_last_error("null pointer argument".to_string());
        return std::ptr::null_mut();
    }

    let wrapper = unsafe { &mut *wrapper };
    let path_str = unsafe { CStr::from_ptr(audio_path) }
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

    let temperature = json_f64(&params, "temperature", 0.0);
    let language = json_str(&params, "language", "en");

    match transcribe(wrapper, path_str, temperature, language) {
        Ok((text, segments)) => {
            let c_text = CString::new(text).unwrap_or_default();

            let c_segments: Vec<WhisperSegment> = segments
                .into_iter()
                .map(|(text, start, end)| {
                    let c_seg_text = CString::new(text).unwrap_or_default();
                    WhisperSegment {
                        text: c_seg_text.into_raw(),
                        start,
                        end,
                    }
                })
                .collect();

            let seg_count = c_segments.len();
            let mut seg_boxed = c_segments.into_boxed_slice();
            let seg_ptr = seg_boxed.as_mut_ptr();
            std::mem::forget(seg_boxed);

            let result = WhisperResult {
                text: c_text.into_raw(),
                segments: seg_ptr,
                segment_count: seg_count,
            };
            Box::into_raw(Box::new(result))
        }
        Err(e) => {
            set_last_error(format!("transcription failed: {e}"));
            std::ptr::null_mut()
        }
    }
}

#[no_mangle]
pub extern "C" fn free_whisper_pipeline(wrapper: *mut WhisperPipelineWrapper) {
    if !wrapper.is_null() {
        unsafe {
            drop(Box::from_raw(wrapper));
        }
    }
}

#[no_mangle]
pub extern "C" fn free_whisper_result(result: *mut WhisperResult) {
    if !result.is_null() {
        unsafe {
            let r = Box::from_raw(result);
            if !r.text.is_null() {
                drop(CString::from_raw(r.text));
            }
            if !r.segments.is_null() {
                let segments =
                    Vec::from_raw_parts(r.segments, r.segment_count, r.segment_count);
                for seg in segments {
                    if !seg.text.is_null() {
                        drop(CString::from_raw(seg.text));
                    }
                }
            }
        }
    }
}
