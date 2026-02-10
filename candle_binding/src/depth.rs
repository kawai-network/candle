use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::sync::Arc;

use candle_core::{DType, Device, Module, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::depth_anything_v2::{DepthAnythingV2, DepthAnythingV2Config};
use candle_transformers::models::dinov2;

use crate::{json_str, parse_config_json, set_last_error, create_hf_repo};
use crate::image_utils;

const DINO_IMG_SIZE: usize = 518;

pub struct DepthPipelineWrapper {
    model: DepthAnythingV2,
    device: Device,
}

/// Depth estimation result: a flat f32 array with (height, width).
#[repr(C)]
pub struct DepthResult {
    pub data: *mut f32,
    pub height: usize,
    pub width: usize,
}

fn load_depth_model(
    config: &serde_json::Value,
    device: &Device,
) -> anyhow::Result<DepthAnythingV2> {
    // Depth Anything V2 uses two model files:
    // 1. DINOv2 backbone (e.g. from "lmz/candle-dino-v2")
    // 2. Depth head (e.g. from "jeroenvlek/depth-anything-v2-safetensors")
    // OR user can specify a single model_id if the model bundles both.

    let cache_dir = config
        .get("cache_dir")
        .and_then(|v| v.as_str());

    // DINOv2 backbone
    let dinov2_model_id = json_str(config, "dinov2_model_id", "lmz/candle-dino-v2");
    let dinov2_file = json_str(config, "dinov2_file", "dinov2_vits14.safetensors");
    let dinov2_repo = create_hf_repo(dinov2_model_id, cache_dir)?;
    let dinov2_path = dinov2_repo.get(dinov2_file)?;

    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[dinov2_path], DType::F32, device)? };
    let dino = dinov2::vit_small(vb)?;

    // Depth Anything V2 head
    let depth_model_id = json_str(
        config,
        "model_id",
        "jeroenvlek/depth-anything-v2-safetensors",
    );
    let depth_file = json_str(config, "depth_file", "depth_anything_v2_vits.safetensors");
    let depth_repo = create_hf_repo(depth_model_id, cache_dir)?;
    let depth_path = depth_repo.get(depth_file)?;

    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[depth_path], DType::F32, device)? };

    let da_config = DepthAnythingV2Config::vit_small();
    let model = DepthAnythingV2::new(Arc::new(dino), da_config, vb)?;

    Ok(model)
}

fn estimate_depth(
    wrapper: &DepthPipelineWrapper,
    image_path: &str,
) -> anyhow::Result<(Vec<f32>, usize, usize)> {
    // Load and get original dimensions
    let img = image::ImageReader::open(image_path)
        .map_err(|e| anyhow::anyhow!("failed to open image: {e}"))?
        .decode()
        .map_err(|e| anyhow::anyhow!("failed to decode image: {e}"))?;
    let original_h = img.height() as usize;
    let original_w = img.width() as usize;

    // Resize for model
    let resized = img.resize_to_fill(
        DINO_IMG_SIZE as u32,
        DINO_IMG_SIZE as u32,
        image::imageops::FilterType::Triangle,
    );
    let rgb = resized.to_rgb8();
    let data = rgb.into_raw();
    let tensor = Tensor::from_vec(data, (DINO_IMG_SIZE, DINO_IMG_SIZE, 3), &wrapper.device)?
        .permute((2, 0, 1))?
        .to_dtype(DType::F32)?;

    // Normalize
    let mean = Tensor::new(&image_utils::IMAGENET_MEAN, &wrapper.device)?.reshape((3, 1, 1))?;
    let std = Tensor::new(&image_utils::IMAGENET_STD, &wrapper.device)?.reshape((3, 1, 1))?;
    let tensor = ((tensor / 255.)?.broadcast_sub(&mean)?.broadcast_div(&std))?;
    let tensor = tensor.unsqueeze(0)?;

    // Forward
    let depth = wrapper.model.forward(&tensor)?;

    // Interpolate back to original size
    let depth = depth.interpolate2d(original_h, original_w)?;

    // Normalize depth to 0..1
    let flat: Vec<f32> = depth.flatten_all()?.to_vec1()?;
    let min_val = flat.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_val = flat.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let range = max_val - min_val;
    let normalized: Vec<f32> = if range > 0.0 {
        flat.iter().map(|v| (v - min_val) / range).collect()
    } else {
        flat
    };

    Ok((normalized, original_h, original_w))
}

#[no_mangle]
pub extern "C" fn new_depth_pipeline(
    config_json: *const c_char,
) -> *mut DepthPipelineWrapper {
    let config = match parse_config_json(config_json) {
        Ok(c) => c,
        Err(e) => {
            set_last_error(e);
            return std::ptr::null_mut();
        }
    };

    let device = Device::Cpu;

    match load_depth_model(&config, &device) {
        Ok(model) => {
            let wrapper = DepthPipelineWrapper { model, device };
            Box::into_raw(Box::new(wrapper))
        }
        Err(e) => {
            set_last_error(format!("failed to load depth model: {e}"));
            std::ptr::null_mut()
        }
    }
}

#[no_mangle]
pub extern "C" fn run_depth_estimation(
    wrapper: *mut DepthPipelineWrapper,
    image_path: *const c_char,
) -> *mut DepthResult {
    if wrapper.is_null() || image_path.is_null() {
        set_last_error("null pointer argument".to_string());
        return std::ptr::null_mut();
    }

    let wrapper = unsafe { &*wrapper };
    let path_str = unsafe { CStr::from_ptr(image_path) }
        .to_str()
        .unwrap_or_default();

    match estimate_depth(wrapper, path_str) {
        Ok((data, height, width)) => {
            let mut boxed = data.into_boxed_slice();
            let ptr = boxed.as_mut_ptr();
            std::mem::forget(boxed);

            let result = DepthResult {
                data: ptr,
                height,
                width,
            };
            Box::into_raw(Box::new(result))
        }
        Err(e) => {
            set_last_error(format!("depth estimation failed: {e}"));
            std::ptr::null_mut()
        }
    }
}

#[no_mangle]
pub extern "C" fn free_depth_pipeline(wrapper: *mut DepthPipelineWrapper) {
    if !wrapper.is_null() {
        unsafe {
            drop(Box::from_raw(wrapper));
        }
    }
}

#[no_mangle]
pub extern "C" fn free_depth_result(result: *mut DepthResult) {
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
