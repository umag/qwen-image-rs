//! Device selection. Picks CUDA when the `cuda` feature is built and a GPU is
//! present; otherwise CPU (the default build, used for Mac-side `cargo check`).

use candle_core::Device;

use crate::Result;

/// Best available device: CUDA:0 under the `cuda` feature, else CPU.
pub fn best_device() -> Result<Device> {
    #[cfg(feature = "cuda")]
    {
        match Device::new_cuda(0) {
            Ok(d) => return Ok(d),
            Err(e) => {
                tracing::warn!("CUDA requested but unavailable ({e}); falling back to CPU");
            }
        }
    }
    Ok(Device::Cpu)
}

/// Human-readable device label for logs.
pub fn label(device: &Device) -> &'static str {
    match device {
        Device::Cpu => "cpu",
        Device::Cuda(_) => "cuda",
        Device::Metal(_) => "metal",
    }
}
