use std::ffi::CStr;
use std::os::raw::c_char;

use candle_core::{DType, Device, Module, Tensor};
use candle_nn::VarBuilder;

use crate::{json_str, json_u64, parse_config_json, set_last_error, create_hf_repo, load_weight_files};
use crate::image_utils;

pub struct SegmentationPipelineWrapper {
    model: candle_transformers::models::segformer::SemanticSegmentationModel,
    device: Device,
    image_size: usize,
    num_labels: usize,
}

/// Segmentation result: class ID per pixel.
#[repr(C)]
pub struct SegmentationResult {
    /// Class ID map (int32 per pixel, row-major).
    pub data: *mut i32,
    pub height: usize,
    pub width: usize,
    pub num_labels: usize,
}

fn load_segmentation_model(
    config: &serde_json::Value,
    device: &Device,
) -> anyhow::Result<SegmentationPipelineWrapper> {
    let model_id = json_str(config, "model_id", "");
    if model_id.is_empty() {
        anyhow::bail!("model_id is required");
    }

    let cache_dir = config.get("cache_dir").and_then(|v| v.as_str());
    let repo = create_hf_repo(model_id, cache_dir)?;

    let config_path = repo.get("config.json")?;
    let config_data: serde_json::Value =
        serde_json::from_reader(std::fs::File::open(&config_path)?)?;

    let image_size = config_data
        .get("image_size")
        .and_then(|v| v.as_u64())
        .unwrap_or(512) as usize;

    let num_labels = config_data
        .get("num_labels")
        .and_then(|v| v.as_u64())
        .unwrap_or_else(|| json_u64(config, "num_labels", 150)) as usize;

    let weight_files = load_weight_files(&repo)?;
    let dtype = DType::F32;
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&weight_files, dtype, device)? };

    let cfg: candle_transformers::models::segformer::Config =
        serde_json::from_reader(std::fs::File::open(&config_path)?)?;

    let model = candle_transformers::models::segformer::SemanticSegmentationModel::new(
        &cfg, num_labels, vb,
    )?;

    Ok(SegmentationPipelineWrapper {
        model,
        device: device.clone(),
        image_size,
        num_labels,
    })
}

fn segment_image(
    wrapper: &SegmentationPipelineWrapper,
    image_path: &str,
) -> anyhow::Result<(Vec<i32>, usize, usize)> {
    let img = image_utils::load_image(image_path, wrapper.image_size, &wrapper.device)?;
    let img = img.unsqueeze(0)?;

    let logits = wrapper.model.forward(&img)?;
    // logits shape: (1, num_labels, H, W)
    let logits = logits.squeeze(0)?; // (num_labels, H, W)

    // argmax over class dimension
    let class_map = logits.argmax(0)?; // (H, W)
    let (h, w) = class_map.dims2()?;
    let data: Vec<i32> = class_map
        .flatten_all()?
        .to_dtype(DType::F32)?
        .to_vec1::<f32>()?
        .into_iter()
        .map(|v| v as i32)
        .collect();

    Ok((data, h, w))
}

#[no_mangle]
pub extern "C" fn new_segmentation_pipeline(
    config_json: *const c_char,
) -> *mut SegmentationPipelineWrapper {
    let config = match parse_config_json(config_json) {
        Ok(c) => c,
        Err(e) => {
            set_last_error(e);
            return std::ptr::null_mut();
        }
    };

    let device = Device::Cpu;

    match load_segmentation_model(&config, &device) {
        Ok(wrapper) => Box::into_raw(Box::new(wrapper)),
        Err(e) => {
            set_last_error(format!("failed to load segmentation model: {e}"));
            std::ptr::null_mut()
        }
    }
}

#[no_mangle]
pub extern "C" fn run_segmentation(
    wrapper: *mut SegmentationPipelineWrapper,
    image_path: *const c_char,
) -> *mut SegmentationResult {
    if wrapper.is_null() || image_path.is_null() {
        set_last_error("null pointer argument".to_string());
        return std::ptr::null_mut();
    }

    let wrapper = unsafe { &*wrapper };
    let path_str = unsafe { CStr::from_ptr(image_path) }
        .to_str()
        .unwrap_or_default();

    match segment_image(wrapper, path_str) {
        Ok((data, height, width)) => {
            let mut boxed = data.into_boxed_slice();
            let ptr = boxed.as_mut_ptr();
            std::mem::forget(boxed);

            let result = SegmentationResult {
                data: ptr,
                height,
                width,
                num_labels: wrapper.num_labels,
            };
            Box::into_raw(Box::new(result))
        }
        Err(e) => {
            set_last_error(format!("segmentation failed: {e}"));
            std::ptr::null_mut()
        }
    }
}

#[no_mangle]
pub extern "C" fn free_segmentation_pipeline(wrapper: *mut SegmentationPipelineWrapper) {
    if !wrapper.is_null() {
        unsafe {
            drop(Box::from_raw(wrapper));
        }
    }
}

#[no_mangle]
pub extern "C" fn free_segmentation_result(result: *mut SegmentationResult) {
    if !result.is_null() {
        unsafe {
            let r = Box::from_raw(result);
            if !r.data.is_null() {
                let total = r.height * r.width;
                drop(Vec::from_raw_parts(r.data, total, total));
            }
        }
    }
}
