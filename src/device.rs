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

/// `(free, total)` device memory in bytes (cuMemGetInfo) on CUDA; `None` on
/// CPU or if the query fails.
pub fn free_vram(device: &Device) -> Option<(usize, usize)> {
    #[cfg(feature = "cuda")]
    if let Device::Cuda(d) = device {
        use candle_core::cuda_backend::cudarc::driver::result::mem_get_info;
        d.cuda_stream().context().bind_to_thread().ok()?;
        return mem_get_info().ok();
    }
    let _ = device;
    None
}

/// Human-readable device label for logs.
pub fn label(device: &Device) -> &'static str {
    match device {
        Device::Cpu => "cpu",
        Device::Cuda(_) => "cuda",
        Device::Metal(_) => "metal",
    }
}
