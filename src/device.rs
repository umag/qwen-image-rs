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

/// The default CUDA memory pool of `device` (the one candle's
/// `cuMemAllocAsync` allocations come from), bound to this thread.
#[cfg(feature = "cuda")]
fn default_pool(
    d: &candle_core::CudaDevice,
) -> Option<candle_core::cuda_backend::cudarc::driver::sys::CUmemoryPool> {
    use candle_core::cuda_backend::cudarc::driver::sys;
    let ctx = d.cuda_stream().context().clone();
    ctx.bind_to_thread().ok()?;
    let mut pool: sys::CUmemoryPool = std::ptr::null_mut();
    unsafe { sys::cuDeviceGetDefaultMemPool(&mut pool, ctx.cu_device()) }
        .result()
        .ok()?;
    Some(pool)
}

/// Release threshold set by [`retain_pool`] so far (bytes; process-wide,
/// only ever raised).
#[cfg(feature = "cuda")]
static RETAINED: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// `(RESERVED_MEM_CURRENT, USED_MEM_CURRENT)` of the default pool.
#[cfg(feature = "cuda")]
fn pool_usage(
    pool: candle_core::cuda_backend::cudarc::driver::sys::CUmemoryPool,
) -> Option<(u64, u64)> {
    use candle_core::cuda_backend::cudarc::driver::sys;
    use sys::CUmemPool_attribute as A;
    let get = |attr| -> Option<u64> {
        let mut v: sys::cuuint64_t = 0;
        unsafe { sys::cuMemPoolGetAttribute(pool, attr, (&mut v as *mut u64).cast()) }
            .result()
            .ok()?;
        Some(v)
    };
    Some((
        get(A::CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT)?,
        get(A::CU_MEMPOOL_ATTR_USED_MEM_CURRENT)?,
    ))
}

/// Let the default CUDA memory pool keep up to `extra` bytes of freed memory
/// on top of what is allocated right now, across stream synchronizations
/// (`CU_MEMPOOL_ATTR_RELEASE_THRESHOLD`, default 0; the threshold counts ALL
/// reserved memory, used or not, so it is set to `used + extra`).
///
/// With threshold 0 every sync hands all freed pool memory back to the driver,
/// so the next VAE decode re-maps its whole multi-GB working set: 267
/// `cuMemAllocAsync` calls took 85 ms of a 281 ms 1024² decode and the GPU sat
/// idle waiting on them. The threshold is bounded (the caller passes its
/// working-set estimate) and only ever raised; `QIR_POOL_RETAIN=0` opts out.
/// No-op on CPU or if the driver call fails.
pub fn retain_pool(device: &Device, extra: usize) {
    #[cfg(feature = "cuda")]
    if let Device::Cuda(d) = device {
        use candle_core::cuda_backend::cudarc::driver::sys;
        use std::sync::atomic::Ordering;
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if !*ON.get_or_init(|| std::env::var("QIR_POOL_RETAIN").map_or(true, |v| v != "0")) {
            return;
        }
        let Some(pool) = default_pool(d) else { return };
        let Some((_, used)) = pool_usage(pool) else {
            return;
        };
        let want = (used as usize).saturating_add(extra);
        if want <= RETAINED.load(Ordering::Relaxed) {
            return;
        }
        let mut v: sys::cuuint64_t = want as u64;
        let attr = sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_RELEASE_THRESHOLD;
        let ok = unsafe { sys::cuMemPoolSetAttribute(pool, attr, (&mut v as *mut u64).cast()) }
            .result()
            .is_ok();
        if ok {
            RETAINED.fetch_max(want, Ordering::Relaxed);
            tracing::debug!(
                threshold_mib = want >> 20,
                used_mib = used >> 20,
                "cuda mempool release threshold"
            );
        }
    }
    let _ = (device, extra);
}

/// Freed-but-retained bytes in the default pool (`RESERVED - USED`), which a
/// new allocation reuses but cuMemGetInfo does not count as free. 0 on CPU,
/// on a failed query, or while [`retain_pool`] has never run.
pub fn pool_slack(device: &Device) -> usize {
    #[cfg(feature = "cuda")]
    if let Device::Cuda(d) = device {
        if RETAINED.load(std::sync::atomic::Ordering::Relaxed) == 0 {
            return 0;
        }
        let Some(pool) = default_pool(d) else {
            return 0;
        };
        let Some((res, used)) = pool_usage(pool) else {
            return 0;
        };
        return res.saturating_sub(used) as usize;
    }
    let _ = device;
    0
}

/// Human-readable device label for logs.
pub fn label(device: &Device) -> &'static str {
    match device {
        Device::Cpu => "cpu",
        Device::Cuda(_) => "cuda",
        Device::Metal(_) => "metal",
    }
}
