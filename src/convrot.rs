//! ConvRot INT8 GEMM bridge: a candle `CustomOp2` that launches the stock-CUTLASS
//! `int8_gemm_s32` kernel (kernels/convrot/int8_gemm.cu). candle has no I8 dtype,
//! so int8 operands ride in **U8** tensors (identical bytes); the output is I32.
//! Only compiled under the `convrot` feature (which implies `cuda`).

use candle_core::backend::BackendStorage;
use candle_core::cuda_backend::cudarc::driver::DevicePtr;
use candle_core::{CpuStorage, CudaStorage, Layout, Shape, Tensor};

use crate::Result;

extern "C" {
    fn int8_gemm_s32(
        c: *mut i32,
        a: *const i8,
        b: *const i8,
        m: i32,
        n: i32,
        k: i32,
        stream: *mut std::ffi::c_void,
    ) -> i32;
}

struct Int8Gemm;

impl candle_core::CustomOp2 for Int8Gemm {
    fn name(&self) -> &'static str {
        "int8-gemm"
    }

    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> candle_core::Result<(CpuStorage, Shape)> {
        candle_core::bail!("int8-gemm is CUDA-only")
    }

    fn cuda_fwd(
        &self,
        a: &CudaStorage,
        a_l: &Layout,
        b: &CudaStorage,
        b_l: &Layout,
    ) -> candle_core::Result<(CudaStorage, Shape)> {
        let dev = a.device().clone();
        let (m, k) = a_l.shape().dims2()?;
        let (n, k2) = b_l.shape().dims2()?;
        if k != k2 {
            candle_core::bail!("int8-gemm K mismatch: {k} vs {k2}");
        }
        // U8 storage holding int8 bytes.
        let a = a.as_cuda_slice::<u8>()?;
        let b = b.as_cuda_slice::<u8>()?;
        let stream = dev.cuda_stream();
        let dst = unsafe { dev.alloc::<i32>(m * n)? };
        // Scope the device_ptr guards so their borrows of `dst` end before the
        // wrap_cuda_slice move below.
        {
            let (a_ptr, _ga) = a.device_ptr(&stream);
            let (b_ptr, _gb) = b.device_ptr(&stream);
            let (c_ptr, _gc) = dst.device_ptr(&stream);
            let rc = unsafe {
                int8_gemm_s32(
                    c_ptr as *mut i32,
                    a_ptr as *const i8,
                    b_ptr as *const i8,
                    m as i32,
                    n as i32,
                    k as i32,
                    stream.cu_stream() as *mut std::ffi::c_void,
                )
            };
            if rc != 0 {
                candle_core::bail!("int8_gemm_s32 failed, rc={rc}");
            }
        }
        let dst = CudaStorage::wrap_cuda_slice(dst, dev);
        Ok((dst, (m, n).into()))
    }
}

/// INT8 GEMM: `A (M,K) u8=int8` @ `Bᵀ` where `B (N,K) u8=int8` -> `C (M,N) i32`.
pub fn int8_gemm(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    Ok(a.apply_op2(b, Int8Gemm)?)
}

extern "C" {
    fn quantize_rows_i8(
        out: *mut u8,
        x: *const std::ffi::c_void,
        inv_scale: *const f32,
        m: i32,
        k: i32,
        stream: *mut std::ffi::c_void,
    );
    fn dequant_i32_bf16(
        out: *mut std::ffi::c_void,
        c: *const i32,
        row_scale: *const f32,
        col_scale: *const f32,
        m: i32,
        n: i32,
        stream: *mut std::ffi::c_void,
    );
}

/// Per-row INT8 quantize: `x (M,K) bf16`, `inv_scale (M,) f32` -> `u8 (M,K)` int8.
struct QuantizeRows;
impl candle_core::CustomOp2 for QuantizeRows {
    fn name(&self) -> &'static str {
        "quantize-rows-i8"
    }
    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> candle_core::Result<(CpuStorage, Shape)> {
        candle_core::bail!("cuda-only")
    }
    fn cuda_fwd(
        &self,
        x: &CudaStorage,
        x_l: &Layout,
        s: &CudaStorage,
        _s_l: &Layout,
    ) -> candle_core::Result<(CudaStorage, Shape)> {
        let dev = x.device().clone();
        let (m, k) = x_l.shape().dims2()?;
        let x = x.as_cuda_slice::<half::bf16>()?;
        let s = s.as_cuda_slice::<f32>()?;
        let stream = dev.cuda_stream();
        let out = unsafe { dev.alloc::<u8>(m * k)? };
        {
            let (xp, _a) = x.device_ptr(&stream);
            let (sp, _b) = s.device_ptr(&stream);
            let (op, _c) = out.device_ptr(&stream);
            unsafe {
                quantize_rows_i8(
                    op as *mut u8,
                    xp as *const std::ffi::c_void,
                    sp as *const f32,
                    m as i32,
                    k as i32,
                    stream.cu_stream() as *mut std::ffi::c_void,
                );
            }
        }
        Ok((CudaStorage::wrap_cuda_slice(out, dev), (m, k).into()))
    }
}

/// Dequant: `c (M,N) i32`, `row_scale (M,)`, `col_scale (N,)` -> `bf16 (M,N)`.
struct Dequant;
impl candle_core::CustomOp3 for Dequant {
    fn name(&self) -> &'static str {
        "dequant-i32-bf16"
    }
    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> candle_core::Result<(CpuStorage, Shape)> {
        candle_core::bail!("cuda-only")
    }
    fn cuda_fwd(
        &self,
        c: &CudaStorage,
        c_l: &Layout,
        rs: &CudaStorage,
        _: &Layout,
        cs: &CudaStorage,
        _: &Layout,
    ) -> candle_core::Result<(CudaStorage, Shape)> {
        let dev = c.device().clone();
        let (m, n) = c_l.shape().dims2()?;
        let c = c.as_cuda_slice::<i32>()?;
        let rs = rs.as_cuda_slice::<f32>()?;
        let cs = cs.as_cuda_slice::<f32>()?;
        let stream = dev.cuda_stream();
        let out = unsafe { dev.alloc::<half::bf16>(m * n)? };
        {
            let (cp, _a) = c.device_ptr(&stream);
            let (rp, _b) = rs.device_ptr(&stream);
            let (sp, _c) = cs.device_ptr(&stream);
            let (op, _d) = out.device_ptr(&stream);
            unsafe {
                dequant_i32_bf16(
                    op as *mut std::ffi::c_void,
                    cp as *const i32,
                    rp as *const f32,
                    sp as *const f32,
                    m as i32,
                    n as i32,
                    stream.cu_stream() as *mut std::ffi::c_void,
                );
            }
        }
        Ok((CudaStorage::wrap_cuda_slice(out, dev), (m, n).into()))
    }
}

use candle_core::D;

/// A ConvRot W8A8 linear: rotated INT8 weight + per-channel scale. Forward
/// rotates + per-token INT8-quantizes the activation, runs the INT8 GEMM, and
/// dequantizes. Output-equivalent to `x Wᵀ` up to INT8 error.
pub struct ConvRotLinear {
    w_i8: Tensor,      // (N, K) u8 holding int8
    col_scale: Tensor, // (N,) f32
    hadamard: Tensor,  // (256, 256) f32 rotation
}

impl ConvRotLinear {
    /// Build from a bf16 `weight (N, K)` and the rotation `r`.
    pub fn from_weight(weight: &Tensor, r: &Tensor) -> Result<Self> {
        let n = weight.dim(0)?;
        let wr = crate::model::rotation::fold_weight(weight, r)?; // (N,K) bf16
        let amax = wr.abs()?.max_keepdim(D::Minus1)?; // (N,1)
        let col_scale = (amax.to_dtype(candle_core::DType::F32)? / 127.0)?.reshape(n)?;
        let inv = (amax.recip()? * 127.0)?
            .to_dtype(candle_core::DType::F32)?
            .reshape(n)?;
        let w_i8 = wr.apply_op2(&inv, QuantizeRows)?; // (N,K) u8
        Ok(Self { w_i8, col_scale, hadamard: r.clone() })
    }

    /// Build from already-rotated-and-quantized tensors (loaded from a
    /// pre-quantized weight file), skipping the rotate+quant done by
    /// `from_weight`. `w_i8 (N,K)` is U8 holding int8; `col_scale (N,)` is f32.
    pub fn from_prequantized(w_i8: Tensor, col_scale: Tensor, r: &Tensor) -> Result<Self> {
        Ok(Self {
            w_i8,
            col_scale,
            hadamard: r.clone(),
        })
    }

    /// The stored INT8 weight and per-column scale, for serialization.
    pub fn export(&self) -> (&Tensor, &Tensor) {
        (&self.w_i8, &self.col_scale)
    }

    /// Forward on `x (..., K)` bf16 -> `(..., N)` bf16.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let dims = x.dims().to_vec();
        let k = *dims.last().unwrap();
        let m: usize = dims[..dims.len() - 1].iter().product();
        let x2 = x.reshape((m, k))?;
        let xr = crate::model::rotation::rotate(&x2, &self.hadamard)?; // (M,K) bf16
        let amax = xr.abs()?.max_keepdim(D::Minus1)?; // (M,1)
        let row_scale = (amax.to_dtype(candle_core::DType::F32)? / 127.0)?.reshape(m)?;
        let inv = (amax.recip()? * 127.0)?
            .to_dtype(candle_core::DType::F32)?
            .reshape(m)?;
        let x_i8 = xr.apply_op2(&inv, QuantizeRows)?; // (M,K) u8
        let c = int8_gemm(&x_i8, &self.w_i8)?; // (M,N) i32
        let n = self.col_scale.dim(0)?;
        let out = c.apply_op3(&row_scale, &self.col_scale, Dequant)?; // (M,N) bf16
        let mut out_dims = dims[..dims.len() - 1].to_vec();
        out_dims.push(n);
        Ok(out.reshape(out_dims)?)
    }
}

/// Validate a ConvRot INT8 linear vs its bf16 reference: cosine similarity.
pub fn self_test_linear() -> Result<f32> {
    use candle_core::{DType, Device};
    let dev = Device::new_cuda(0)?;
    let (m, n, k) = (16usize, 512usize, 512usize); // K multiple of 256
    let w = Tensor::randn(0f32, 1f32, (n, k), &dev)?.to_dtype(DType::BF16)?;
    let x = Tensor::randn(0f32, 1f32, (m, k), &dev)?.to_dtype(DType::BF16)?;
    let r = crate::model::rotation::regular_hadamard_256(&dev)?;
    let cr = ConvRotLinear::from_weight(&w, &r)?;
    let y_int8 = cr.forward(&x)?.to_dtype(DType::F32)?;
    let y_ref = x.to_dtype(DType::F32)?.matmul(&w.to_dtype(DType::F32)?.t()?)?;
    let dot = (&y_int8 * &y_ref)?.sum_all()?.to_scalar::<f32>()?;
    let na = y_int8.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt();
    let nb = y_ref.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt();
    Ok(dot / (na * nb + 1e-8))
}

/// Time ConvRot INT8 vs bf16 at the DiT MLP shape (M=4117, K=4096, N=12288).
/// Returns (convrot_ms, bf16_ms) per forward.
pub fn bench_linear(iters: usize) -> Result<(f64, f64)> {
    use candle_core::{DType, Device};
    let dev = Device::new_cuda(0)?;
    let (m, n, k) = (4117usize, 12288usize, 4096usize);
    let w = Tensor::randn(0f32, 1f32, (n, k), &dev)?.to_dtype(DType::BF16)?;
    let x = Tensor::randn(0f32, 1f32, (m, k), &dev)?.to_dtype(DType::BF16)?;
    let r = crate::model::rotation::regular_hadamard_256(&dev)?;
    let cr = ConvRotLinear::from_weight(&w, &r)?;
    let wt = w.t()?.contiguous()?;
    let time = |f: &dyn Fn() -> Result<()>| -> Result<f64> {
        f()?;
        dev.synchronize()?;
        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            f()?;
        }
        dev.synchronize()?;
        Ok(t0.elapsed().as_secs_f64() * 1e3 / iters as f64)
    };
    let convrot_ms = time(&|| {
        cr.forward(&x)?;
        Ok(())
    })?;
    let bf16_ms = time(&|| {
        x.matmul(&wt)?;
        Ok(())
    })?;
    Ok((convrot_ms, bf16_ms))
}

/// End-to-end bridge self-test: candle U8(int8) tensors -> kernel -> candle I32,
/// compared to a CPU int reference. Only the kernel runs on the GPU (host-side
/// compare avoids needing candle cast kernels). Returns the max abs difference.
pub fn self_test() -> Result<i64> {
    use candle_core::Device;
    let dev = Device::new_cuda(0)?;
    let (m, n, k) = (64usize, 128usize, 256usize);
    let ai: Vec<i8> = (0..m * k).map(|i| ((i * 7) % 15) as i8 - 7).collect();
    let bi: Vec<i8> = (0..n * k).map(|i| ((i * 13) % 15) as i8 - 7).collect();
    let a_u8 = Tensor::from_vec(ai.iter().map(|&v| v as u8).collect::<Vec<u8>>(), (m, k), &dev)?;
    let b_u8 = Tensor::from_vec(bi.iter().map(|&v| v as u8).collect::<Vec<u8>>(), (n, k), &dev)?;
    let c: Vec<i32> = int8_gemm(&a_u8, &b_u8)?.flatten_all()?.to_vec1::<i32>()?;

    let mut maxdiff = 0i64;
    for mi in 0..m {
        for ni in 0..n {
            let mut acc = 0i64;
            for kk in 0..k {
                acc += ai[mi * k + kk] as i64 * bi[ni * k + kk] as i64;
            }
            maxdiff = maxdiff.max((acc - c[mi * n + ni] as i64).abs());
        }
    }
    Ok(maxdiff)
}
