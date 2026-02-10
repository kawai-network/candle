use candle_core::{DType, Device, Result, Tensor};

pub const IMAGENET_MEAN: [f32; 3] = [0.485f32, 0.456, 0.406];
pub const IMAGENET_STD: [f32; 3] = [0.229f32, 0.224, 0.225];

/// Load an image from a file path and preprocess for ImageNet-style models.
/// Returns a tensor with shape (3, res, res) with ImageNet normalization applied.
pub fn load_image(path: &str, res: usize, device: &Device) -> Result<Tensor> {
    load_image_with_std_mean(path, res, &IMAGENET_MEAN, &IMAGENET_STD, device)
}

pub fn load_image_with_std_mean(
    path: &str,
    res: usize,
    mean: &[f32; 3],
    std: &[f32; 3],
    device: &Device,
) -> Result<Tensor> {
    let img = image::ImageReader::open(path)
        .map_err(candle_core::Error::wrap)?
        .decode()
        .map_err(candle_core::Error::wrap)?
        .resize_to_fill(
            res as u32,
            res as u32,
            image::imageops::FilterType::Triangle,
        );
    let img = img.to_rgb8();
    let data = img.into_raw();
    let data = Tensor::from_vec(data, (res, res, 3), device)?.permute((2, 0, 1))?;
    let mean = Tensor::new(mean, device)?.reshape((3, 1, 1))?;
    let std = Tensor::new(std, device)?.reshape((3, 1, 1))?;
    (data.to_dtype(DType::F32)? / 255.)?
        .broadcast_sub(&mean)?
        .broadcast_div(&std)
}

/// Load an image from raw bytes (e.g. PNG/JPEG) and preprocess.
pub fn load_image_from_bytes(
    bytes: &[u8],
    res: usize,
    mean: &[f32; 3],
    std: &[f32; 3],
    device: &Device,
) -> Result<Tensor> {
    let img = image::load_from_memory(bytes)
        .map_err(candle_core::Error::wrap)?
        .resize_to_fill(
            res as u32,
            res as u32,
            image::imageops::FilterType::Triangle,
        );
    let img = img.to_rgb8();
    let data = img.into_raw();
    let data = Tensor::from_vec(data, (res, res, 3), device)?.permute((2, 0, 1))?;
    let mean = Tensor::new(mean, device)?.reshape((3, 1, 1))?;
    let std = Tensor::new(std, device)?.reshape((3, 1, 1))?;
    (data.to_dtype(DType::F32)? / 255.)?
        .broadcast_sub(&mean)?
        .broadcast_div(&std)
}
