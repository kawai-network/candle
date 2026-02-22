use std::cell::RefCell;
use std::ffi::{CStr, CString};
use std::os::raw::c_char;

mod classification;
mod clip;
mod depth;
mod embeddings;
mod image_utils;
mod segmentation;
mod t5;
mod text_generation;
mod translation;
mod video;
mod whisper;

// Re-export FFI functions
pub use classification::*;
pub use clip::*;
pub use depth::*;
pub use embeddings::*;
pub use segmentation::*;
pub use t5::*;
pub use text_generation::*;
pub use translation::*;
pub use video::*;
pub use whisper::*;

// --- Thread-local error handling ---

thread_local! {
    static LAST_ERROR: RefCell<Option<CString>> = RefCell::new(None);
}

/// Set the last error message for the current thread.
pub(crate) fn set_last_error(msg: String) {
    LAST_ERROR.with(|e| {
        *e.borrow_mut() = CString::new(msg).ok();
    });
}

/// Retrieve the last error message. Returns null if no error is set.
/// The returned pointer is valid until the next FFI call on the same thread.
#[no_mangle]
pub extern "C" fn candle_last_error() -> *const c_char {
    LAST_ERROR.with(|e| match e.borrow().as_ref() {
        Some(s) => s.as_ptr(),
        None => std::ptr::null(),
    })
}

/// Return the binding version string.
#[no_mangle]
pub extern "C" fn candle_binding_version() -> *const c_char {
    // Static so the pointer remains valid
    static VERSION: &[u8] = b"0.1.0\0";
    VERSION.as_ptr() as *const c_char
}

// --- Helper: parse JSON config string from C ---

pub(crate) fn parse_config_json(config_json: *const c_char) -> Result<serde_json::Value, String> {
    if config_json.is_null() {
        return Err("config_json is null".to_string());
    }
    let c_str = unsafe { CStr::from_ptr(config_json) };
    let json_str = c_str
        .to_str()
        .map_err(|e| format!("invalid UTF-8 in config: {e}"))?;
    serde_json::from_str(json_str).map_err(|e| format!("invalid JSON config: {e}"))
}

/// Helper: get a string field from a JSON Value, with a default
pub(crate) fn json_str<'a>(val: &'a serde_json::Value, key: &str, default: &'a str) -> &'a str {
    val.get(key).and_then(|v| v.as_str()).unwrap_or(default)
}

/// Helper: get an integer field from a JSON Value, with a default
pub(crate) fn json_u64(val: &serde_json::Value, key: &str, default: u64) -> u64 {
    val.get(key).and_then(|v| v.as_u64()).unwrap_or(default)
}

/// Helper: get a float field from a JSON Value, with a default
pub(crate) fn json_f64(val: &serde_json::Value, key: &str, default: f64) -> f64 {
    val.get(key).and_then(|v| v.as_f64()).unwrap_or(default)
}

/// Helper: get a bool field from a JSON Value, with a default
pub(crate) fn json_bool(val: &serde_json::Value, key: &str, default: bool) -> bool {
    val.get(key).and_then(|v| v.as_bool()).unwrap_or(default)
}

/// Shared helper: load safetensors weight files from an HF repo.
/// Tries single file first, then sharded format.
pub(crate) fn load_weight_files(
    repo: &hf_hub::api::sync::ApiRepo,
) -> anyhow::Result<Vec<std::path::PathBuf>> {
    if let Ok(p) = repo.get("model.safetensors") {
        Ok(vec![p])
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
        Ok(paths)
    }
}

/// Shared helper: create an HF repo handle with optional cache_dir.
pub(crate) fn create_hf_repo(
    model_id: &str,
    cache_dir: Option<&str>,
) -> anyhow::Result<hf_hub::api::sync::ApiRepo> {
    create_hf_repo_with_revision(model_id, cache_dir, None)
}

/// Shared helper: create an HF repo handle with optional cache_dir and revision.
pub(crate) fn create_hf_repo_with_revision(
    model_id: &str,
    cache_dir: Option<&str>,
    revision: Option<&str>,
) -> anyhow::Result<hf_hub::api::sync::ApiRepo> {
    let api = if let Some(dir) = cache_dir {
        hf_hub::api::sync::ApiBuilder::new()
            .with_cache_dir(std::path::PathBuf::from(dir))
            .build()?
    } else {
        hf_hub::api::sync::Api::new()?
    };

    let repo = if let Some(rev) = revision {
        api.repo(hf_hub::Repo::with_revision(
            model_id.to_string(),
            hf_hub::RepoType::Model,
            rev.to_string(),
        ))
    } else {
        api.model(model_id.to_string())
    };
    Ok(repo)
}
