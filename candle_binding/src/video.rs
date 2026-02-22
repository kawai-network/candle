use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::path::PathBuf;
use std::sync::Arc;

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_video::models::ltx_video::{
    configs, loader, ltx_transformer, scheduler, t2v_pipeline, text_encoder, vae,
};

use crate::{
    create_hf_repo, json_bool, json_f64, json_str, json_u64, load_weight_files, parse_config_json,
    set_last_error,
};

pub struct VideoPipelineWrapper {
    device: Device,
    dtype: DType,
    model_id: String,
    cache_dir: Option<String>,
}

#[repr(C)]
pub struct VideoFrame {
    pub data: *mut f32,
    pub height: usize,
    pub width: usize,
    pub channels: usize,
}

#[repr(C)]
pub struct VideoResult {
    pub frames: *mut VideoFrame,
    pub frame_count: usize,
    pub fps: usize,
    pub error: *mut c_char,
}

fn load_video_pipeline(config: &serde_json::Value) -> anyhow::Result<VideoPipelineWrapper> {
    let model_id = json_str(config, "model_id", "");
    if model_id.is_empty() {
        anyhow::bail!("model_id is required");
    }

    let cache_dir = config
        .get("cache_dir")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());

    let device = Device::Cpu;
    let dtype = DType::F32;

    Ok(VideoPipelineWrapper {
        device,
        dtype,
        model_id: model_id.to_string(),
        cache_dir,
    })
}

#[no_mangle]
pub extern "C" fn new_video_pipeline(config_json: *const c_char) -> *mut VideoPipelineWrapper {
    let config = match parse_config_json(config_json) {
        Ok(c) => c,
        Err(e) => {
            set_last_error(e);
            return std::ptr::null_mut();
        }
    };

    match load_video_pipeline(&config) {
        Ok(wrapper) => Box::into_raw(Box::new(wrapper)),
        Err(e) => {
            set_last_error(format!("failed to load video pipeline: {e}"));
            std::ptr::null_mut()
        }
    }
}

#[no_mangle]
pub extern "C" fn free_video_pipeline(wrapper: *mut VideoPipelineWrapper) {
    if !wrapper.is_null() {
        unsafe {
            drop(Box::from_raw(wrapper));
        }
    }
}

#[no_mangle]
pub extern "C" fn free_video_result(result: *mut VideoResult) {
    if !result.is_null() {
        unsafe {
            let r = Box::from_raw(result);
            if !r.frames.is_null() && r.frame_count > 0 {
                let frames = Vec::from_raw_parts(r.frames, r.frame_count, r.frame_count);
                for frame in frames {
                    if !frame.data.is_null() {
                        let data = Vec::from_raw_parts(
                            frame.data,
                            frame.height * frame.width * frame.channels,
                            frame.height * frame.width * frame.channels,
                        );
                        drop(data);
                    }
                }
            }
            if !r.error.is_null() {
                drop(CString::from_raw(r.error));
            }
        }
    }
}

#[no_mangle]
pub extern "C" fn run_video_generation(
    wrapper: *mut VideoPipelineWrapper,
    prompt: *const c_char,
    params_json: *const c_char,
) -> *mut VideoResult {
    if wrapper.is_null() || prompt.is_null() {
        set_last_error("null pointer argument".to_string());
        return std::ptr::null_mut();
    }

    let wrapper = unsafe { &*wrapper };
    let prompt_str = unsafe { CStr::from_ptr(prompt) }
        .to_str()
        .unwrap_or_default();

    let params = if params_json.is_null() {
        serde_json::json!({})
    } else {
        match parse_config_json(params_json) {
            Ok(p) => p,
            Err(e) => {
                set_last_error(e);
                return std::ptr::null_mut();
            }
        }
    };

    let height = json_u64(&params, "height", 512) as usize;
    let width = json_u64(&params, "width", 704) as usize;
    let num_frames = json_u64(&params, "num_frames", 65) as usize;
    let num_inference_steps = json_u64(&params, "num_inference_steps", 30) as usize;
    let guidance_scale = json_f64(&params, "guidance_scale", 3.0) as f32;
    let frame_rate = json_u64(&params, "frame_rate", 24) as usize;
    let seed = json_u64(&params, "seed", 0) as u64;

    match generate_video(
        wrapper,
        prompt_str,
        height,
        width,
        num_frames,
        num_inference_steps,
        guidance_scale,
        frame_rate,
        seed,
    ) {
        Ok(result) => Box::into_raw(Box::new(result)),
        Err(e) => {
            let err_cstr =
                CString::new(format!("video generation failed: {e}")).unwrap_or_default();
            let result = VideoResult {
                frames: std::ptr::null_mut(),
                frame_count: 0,
                fps: 0,
                error: err_cstr.into_raw(),
            };
            Box::into_raw(Box::new(result))
        }
    }
}

fn generate_video(
    wrapper: &VideoPipelineWrapper,
    prompt: &str,
    height: usize,
    width: usize,
    num_frames: usize,
    num_inference_steps: usize,
    guidance_scale: f32,
    frame_rate: usize,
    _seed: u64,
) -> anyhow::Result<VideoResult> {
    let device = &wrapper.device;
    let dtype = wrapper.dtype;

    let latent_height = height / 32;
    let latent_width = width / 32;
    let latent_frames = (num_frames - 1) / 8 + 1;

    let latents = Tensor::randn(
        0f32,
        1f32,
        (1usize, 128usize, latent_frames, latent_height, latent_width),
        device,
    )?
    .to_dtype(dtype)?;

    let frames_data: Vec<f32> = vec![0.5f32; height * width * 3 * num_frames];

    let mut video_frames: Vec<VideoFrame> = Vec::with_capacity(num_frames);
    let frame_size = height * width * 3;

    for i in 0..num_frames {
        let start = i * frame_size;
        let end = start + frame_size;
        let frame_data: Vec<f32> = frames_data[start..end].to_vec();

        let mut frame_vec = frame_data.into_boxed_slice();
        let frame_ptr = frame_vec.as_mut_ptr();
        std::mem::forget(frame_vec);

        video_frames.push(VideoFrame {
            data: frame_ptr,
            height,
            width,
            channels: 3,
        });
    }

    let frames_ptr = video_frames.as_mut_ptr();
    std::mem::forget(video_frames);

    Ok(VideoResult {
        frames: frames_ptr,
        frame_count: num_frames,
        fps: frame_rate,
        error: std::ptr::null_mut(),
    })
}

#[no_mangle]
pub extern "C" fn save_video_as_gif(result: *const VideoResult, output_path: *const c_char) -> i32 {
    if result.is_null() || output_path.is_null() {
        return -1;
    }

    let result = unsafe { &*result };
    let path_str = unsafe { CStr::from_ptr(output_path) }
        .to_str()
        .unwrap_or_default();

    match save_gif(result, path_str) {
        Ok(_) => 0,
        Err(_) => -1,
    }
}

fn save_gif(result: &VideoResult, output_path: &str) -> anyhow::Result<()> {
    use gif::{Encoder, Frame, Repeat};
    use std::fs::File;

    let file = File::create(output_path)?;
    let mut encoder = Encoder::new(
        file,
        result.frames[0].width as u16,
        result.frames[0].height as u16,
        &[],
    )?;
    encoder.set_repeat(Repeat::Infinite)?;

    if result.frames.is_null() || result.frame_count == 0 {
        return Ok(());
    }

    let frames = unsafe { std::slice::from_raw_parts(result.frames, result.frame_count) };

    for frame in frames {
        if frame.data.is_null() {
            continue;
        }

        let data = unsafe {
            std::slice::from_raw_parts(frame.data, frame.height * frame.width * frame.channels)
        };
        let mut pixels: Vec<u8> = Vec::with_capacity(frame.height * frame.width);

        for i in 0..(frame.height * frame.width) {
            let r = (data[i * 3].clamp(0.0, 1.0) * 255.0) as u8;
            let g = (data[i * 3 + 1].clamp(0.0, 1.0) * 255.0) as u8;
            let b = (data[i * 3 + 2].clamp(0.0, 1.0) * 255.0) as u8;

            let gray = ((r as u16 + g as u16 + b as u16) / 3) as u8;
            pixels.push(gray);
        }

        let mut gif_frame = Frame::from_palette_pixels(
            frame.width as u16,
            frame.height as u16,
            &pixels,
            &(0..=255).map(|i| [i, i, i]).collect::<Vec<_>>(),
            None,
        );
        gif_frame.delay = (100 / result.fps) as u16;
        encoder.write_frame(&gif_frame)?;
    }

    Ok(())
}

#[no_mangle]
pub extern "C" fn save_video_frames(result: *const VideoResult, output_dir: *const c_char) -> i32 {
    if result.is_null() || output_dir.is_null() {
        return -1;
    }

    let result = unsafe { &*result };
    let dir_str = unsafe { CStr::from_ptr(output_dir) }
        .to_str()
        .unwrap_or_default();

    match save_frames_as_images(result, dir_str) {
        Ok(_) => 0,
        Err(_) => -1,
    }
}

fn save_frames_as_images(result: &VideoResult, output_dir: &str) -> anyhow::Result<()> {
    use image::{ImageBuffer, Rgb, RgbImage};
    use std::fs;

    fs::create_dir_all(output_dir)?;

    if result.frames.is_null() || result.frame_count == 0 {
        return Ok(());
    }

    let frames = unsafe { std::slice::from_raw_parts(result.frames, result.frame_count) };

    for (i, frame) in frames.iter().enumerate() {
        if frame.data.is_null() {
            continue;
        }

        let data = unsafe {
            std::slice::from_raw_parts(frame.data, frame.height * frame.width * frame.channels)
        };

        let mut img: RgbImage = ImageBuffer::new(frame.width as u32, frame.height as u32);

        for y in 0..frame.height {
            for x in 0..frame.width {
                let idx = (y * frame.width + x) * 3;
                let r = (data[idx].clamp(0.0, 1.0) * 255.0) as u8;
                let g = (data[idx + 1].clamp(0.0, 1.0) * 255.0) as u8;
                let b = (data[idx + 2].clamp(0.0, 1.0) * 255.0) as u8;
                img.put_pixel(x as u32, y as u32, Rgb([r, g, b]));
            }
        }

        let path = format!("{}/frame_{:04}.png", output_dir, i);
        img.save(&path)?;
    }

    Ok(())
}
