use std::ffi::{CStr, CString};
use std::os::raw::c_char;

use candle_core::{DType, Device, Module, Tensor};
use candle_nn::VarBuilder;

use crate::{json_str, json_u64, parse_config_json, set_last_error, create_hf_repo, load_weight_files};
use crate::image_utils;

/// Opaque wrapper for a classification pipeline.
pub struct ClassificationPipelineWrapper {
    model: Box<dyn ClassificationModel>,
    device: Device,
    image_size: usize,
    id2label: Vec<String>,
}

trait ClassificationModel: Send {
    fn forward(&self, input: &Tensor) -> anyhow::Result<Tensor>;
}

// --- ViT ---
struct ViTModel {
    model: candle_transformers::models::vit::Model,
}

impl ClassificationModel for ViTModel {
    fn forward(&self, input: &Tensor) -> anyhow::Result<Tensor> {
        Ok(self.model.forward(input)?)
    }
}

// --- SegFormer classification ---
struct SegFormerClassModel {
    model: candle_transformers::models::segformer::ImageClassificationModel,
}

impl ClassificationModel for SegFormerClassModel {
    fn forward(&self, input: &Tensor) -> anyhow::Result<Tensor> {
        Ok(self.model.forward(input)?)
    }
}

/// Single classification prediction.
#[repr(C)]
pub struct ClassificationPrediction {
    pub label: *mut c_char,
    pub score: f32,
}

/// Result of image classification.
#[repr(C)]
pub struct ClassificationResult {
    pub predictions: *mut ClassificationPrediction,
    pub count: usize,
}

fn load_classification_model(
    config: &serde_json::Value,
    device: &Device,
) -> anyhow::Result<(Box<dyn ClassificationModel>, usize, Vec<String>)> {
    let model_id = json_str(config, "model_id", "");
    if model_id.is_empty() {
        anyhow::bail!("model_id is required");
    }

    let cache_dir = config
        .get("cache_dir")
        .and_then(|v| v.as_str());

    let repo = create_hf_repo(model_id, cache_dir)?;

    let config_path = repo.get("config.json")?;
    let config_data: serde_json::Value =
        serde_json::from_reader(std::fs::File::open(&config_path)?)?;

    let model_type = config_data
        .get("model_type")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_lowercase();

    // Extract id2label mapping from config.json
    let id2label = extract_id2label(&config_data);
    let num_labels = if id2label.is_empty() {
        json_u64(config, "num_labels", 1000) as usize
    } else {
        id2label.len()
    };

    // Determine image_size from config
    let image_size = config_data
        .get("image_size")
        .and_then(|v| v.as_u64())
        .unwrap_or(224) as usize;

    // Load weights
    let weight_files = load_weight_files(&repo)?;
    let dtype = DType::F32;
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&weight_files, dtype, device)? };

    let model: Box<dyn ClassificationModel> =
        if model_type == "vit" || model_type.contains("vit") {
            let cfg: candle_transformers::models::vit::Config =
                serde_json::from_reader(std::fs::File::open(&config_path)?)?;
            let m = candle_transformers::models::vit::Model::new(&cfg, num_labels, vb)?;
            Box::new(ViTModel { model: m })
        } else if model_type == "segformer" {
            let cfg: candle_transformers::models::segformer::Config =
                serde_json::from_reader(std::fs::File::open(&config_path)?)?;
            let m = candle_transformers::models::segformer::ImageClassificationModel::new(&cfg, num_labels, vb)?;
            Box::new(SegFormerClassModel { model: m })
        } else {
            anyhow::bail!("unsupported classification model_type: {}", model_type);
        };

    Ok((model, image_size, id2label))
}

fn extract_id2label(config_data: &serde_json::Value) -> Vec<String> {
    if let Some(map) = config_data.get("id2label").and_then(|v| v.as_object()) {
        let mut labels: Vec<(usize, String)> = map
            .iter()
            .filter_map(|(k, v)| {
                let idx = k.parse::<usize>().ok()?;
                let label = v.as_str()?.to_string();
                Some((idx, label))
            })
            .collect();
        labels.sort_by_key(|(idx, _)| *idx);
        labels.into_iter().map(|(_, label)| label).collect()
    } else {
        Vec::new()
    }
}

fn classify_image(
    wrapper: &ClassificationPipelineWrapper,
    image_path: &str,
    top_k: usize,
) -> anyhow::Result<Vec<(String, f32)>> {
    let img = image_utils::load_image(image_path, wrapper.image_size, &wrapper.device)?;
    let img = img.unsqueeze(0)?; // Add batch dim

    let logits = wrapper.model.forward(&img)?;
    let logits = logits.squeeze(0)?;

    // Softmax
    let probs = candle_nn::ops::softmax(&logits, 0)?;
    let probs_vec: Vec<f32> = probs.to_vec1()?;

    // Get top-k
    let mut indexed: Vec<(usize, f32)> = probs_vec.into_iter().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    indexed.truncate(top_k);

    let results: Vec<(String, f32)> = indexed
        .into_iter()
        .map(|(idx, score)| {
            let label = if idx < wrapper.id2label.len() {
                wrapper.id2label[idx].clone()
            } else {
                format!("class_{}", idx)
            };
            (label, score)
        })
        .collect();

    Ok(results)
}

#[no_mangle]
pub extern "C" fn new_classification_pipeline(
    config_json: *const c_char,
) -> *mut ClassificationPipelineWrapper {
    let config = match parse_config_json(config_json) {
        Ok(c) => c,
        Err(e) => {
            set_last_error(e);
            return std::ptr::null_mut();
        }
    };

    let device = Device::Cpu;

    match load_classification_model(&config, &device) {
        Ok((model, image_size, id2label)) => {
            let wrapper = ClassificationPipelineWrapper {
                model,
                device,
                image_size,
                id2label,
            };
            Box::into_raw(Box::new(wrapper))
        }
        Err(e) => {
            set_last_error(format!("failed to load classification model: {e}"));
            std::ptr::null_mut()
        }
    }
}

#[no_mangle]
pub extern "C" fn run_classification(
    wrapper: *mut ClassificationPipelineWrapper,
    image_path: *const c_char,
    top_k: usize,
) -> *mut ClassificationResult {
    if wrapper.is_null() || image_path.is_null() {
        set_last_error("null pointer argument".to_string());
        return std::ptr::null_mut();
    }

    let wrapper = unsafe { &*wrapper };
    let path_str = unsafe { CStr::from_ptr(image_path) }
        .to_str()
        .unwrap_or_default();

    let top_k = if top_k == 0 { 5 } else { top_k };

    match classify_image(wrapper, path_str, top_k) {
        Ok(predictions) => {
            let count = predictions.len();
            let mut c_preds: Vec<ClassificationPrediction> = predictions
                .into_iter()
                .map(|(label, score)| {
                    let c_label = CString::new(label).unwrap_or_default();
                    ClassificationPrediction {
                        label: c_label.into_raw(),
                        score,
                    }
                })
                .collect();

            let ptr = c_preds.as_mut_ptr();
            std::mem::forget(c_preds);

            let result = ClassificationResult {
                predictions: ptr,
                count,
            };
            Box::into_raw(Box::new(result))
        }
        Err(e) => {
            set_last_error(format!("classification failed: {e}"));
            std::ptr::null_mut()
        }
    }
}

#[no_mangle]
pub extern "C" fn free_classification_pipeline(wrapper: *mut ClassificationPipelineWrapper) {
    if !wrapper.is_null() {
        unsafe {
            drop(Box::from_raw(wrapper));
        }
    }
}

#[no_mangle]
pub extern "C" fn free_classification_result(result: *mut ClassificationResult) {
    if !result.is_null() {
        unsafe {
            let r = Box::from_raw(result);
            if !r.predictions.is_null() {
                let preds = Vec::from_raw_parts(r.predictions, r.count, r.count);
                for pred in preds {
                    if !pred.label.is_null() {
                        drop(CString::from_raw(pred.label));
                    }
                }
            }
        }
    }
}
