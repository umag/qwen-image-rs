//! ConvRot INT8 GEMM bridge: a candle `CustomOp2` that launches the stock-CUTLASS
//! `int8_gemm_s32` kernel (kernels/convrot/int8_gemm.cu). candle has no I8 dtype,
//! so int8 operands ride in **U8** tensors (identical bytes); the output is I32.
//! Only compiled under the `convrot` feature (which implies `cuda`).

use candle_core::backend::BackendStorage;
use candle_core::cuda_backend::cudarc::driver::DevicePtr;
use candle_core::{CpuStorage, CudaStorage, Layout, Shape, Tensor};

use crate::layout::dense_byte_offset;
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
    // Fused INT8 GEMM + per-row/per-col dequant epilogue -> bf16.
    // d[m,n] = bf16(acc[m,n] * s_row[m] * s_col[n]).
    fn int8_gemm_dequant_bf16(
        d: *mut std::ffi::c_void,
        a: *const i8,
        b: *const i8,
        s_row: *const f32,
        s_col: *const f32,
        m: i32,
        n: i32,
        k: i32,
        stream: *mut std::ffi::c_void,
    ) -> i32;
    // Same fused GEMM, f16 store: d[m,n] = f16(acc[m,n] * s_row[m] * s_col[n]).
    fn int8_gemm_dequant_f16(
        d: *mut std::ffi::c_void,
        a: *const i8,
        b: *const i8,
        s_row: *const f32,
        s_col: *const f32,
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
        let ao = dense_byte_offset::<u8>(a_l, "int8-gemm a")?;
        let bo = dense_byte_offset::<u8>(b_l, "int8-gemm b")?;
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
            let (a_ptr, b_ptr) = (a_ptr + ao, b_ptr + bo);
            if !a_ptr.is_multiple_of(16) || !b_ptr.is_multiple_of(16) {
                candle_core::bail!("int8-gemm: misaligned view (a {a_ptr:#x}, b {b_ptr:#x})");
            }
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
    fn quantize_rows_fused_launch(
        out: *mut u8,
        x: *const std::ffi::c_void,
        row_scale: *mut f32,
        m: i32,
        k: i32,
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
        s_l: &Layout,
    ) -> candle_core::Result<(CudaStorage, Shape)> {
        let dev = x.device().clone();
        let (m, k) = x_l.shape().dims2()?;
        let xo = dense_byte_offset::<half::bf16>(x_l, "quantize-rows x")?;
        let so = dense_byte_offset::<f32>(s_l, "quantize-rows inv_scale")?;
        let x = x.as_cuda_slice::<half::bf16>()?;
        let s = s.as_cuda_slice::<f32>()?;
        let stream = dev.cuda_stream();
        let out = unsafe { dev.alloc::<u8>(m * k)? };
        {
            let (xp, _a) = x.device_ptr(&stream);
            let (sp, _b) = s.device_ptr(&stream);
            let (op, _c) = out.device_ptr(&stream);
            let (xp, sp) = (xp + xo, sp + so);
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

/// Fused per-row activation quantize: `x (M,K) bf16` -> int8 `u8 (M,K)`, while
/// writing the per-row scale (amax/127) IN PLACE into the pre-allocated
/// `row_scale (M,) f32` second input. One CTA-per-row kernel does the
/// `max(|x|)` reduction and the quantize in a single pass, replacing candle's
/// abs + max_keepdim + recip. The in-place write is safe: `row_scale` is a
/// fresh zeros buffer used only after this op, no aliasing, no autograd.
struct QuantizeFused;
impl candle_core::CustomOp2 for QuantizeFused {
    fn name(&self) -> &'static str {
        "quantize-rows-fused"
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
        rs: &CudaStorage,
        rs_l: &Layout,
    ) -> candle_core::Result<(CudaStorage, Shape)> {
        let dev = x.device().clone();
        let (m, k) = x_l.shape().dims2()?;
        let xo = dense_byte_offset::<half::bf16>(x_l, "quantize-rows-fused x")?;
        let ro = dense_byte_offset::<f32>(rs_l, "quantize-rows-fused row_scale")?;
        let x = x.as_cuda_slice::<half::bf16>()?;
        let rs = rs.as_cuda_slice::<f32>()?; // pre-allocated (m,), written in place
        let stream = dev.cuda_stream();
        let out = unsafe { dev.alloc::<u8>(m * k)? };
        {
            let (xp, _a) = x.device_ptr(&stream);
            let (rp, _b) = rs.device_ptr(&stream);
            let (op, _c) = out.device_ptr(&stream);
            let (xp, rp) = (xp + xo, rp + ro);
            unsafe {
                quantize_rows_fused_launch(
                    op as *mut u8,
                    xp as *const std::ffi::c_void,
                    rp as *mut f32,
                    m as i32,
                    k as i32,
                    stream.cu_stream() as *mut std::ffi::c_void,
                );
            }
        }
        Ok((CudaStorage::wrap_cuda_slice(out, dev), (m, k).into()))
    }
}

/// Fused activation quantize: `x (M,K) bf16` -> (`int8 (M,K) u8`, `row_scale (M,) f32`)
/// in one kernel pass. `row_scale[m] = max(|x[m,:]|)/127`.
fn quantize_rows_fused(x: &Tensor) -> Result<(Tensor, Tensor)> {
    let (m, _k) = x.dims2()?;
    let row_scale = Tensor::zeros(m, candle_core::DType::F32, x.device())?;
    let x_i8 = x.apply_op2(&row_scale, QuantizeFused)?;
    Ok((x_i8, row_scale))
}

/// Output precision of the fused dequant epilogue: the f32 epilogue value is
/// rounded once to this 16-bit type on store.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EpilogueOut {
    Bf16,
    /// f16 — for SageAttention's FP16 P·V operand (V), so it needs no cast.
    F16,
}

/// Fused INT8 GEMM + per-row/per-col dequant epilogue.
///
/// `a (M,K) u8=int8` @ `Bᵀ` where `b (N,K) u8=int8`, then the CUTLASS epilogue
/// applies `out[m,n] = acc[m,n] * s_row[m] * s_col[n]` and rounds to the
/// requested 16-bit type (bf16 or f16) — producing `(M,N)` directly, with no
/// i32 intermediate and no separate dequant or cast kernel.
///
/// candle has no `CustomOp4`, so the two scale vectors ride in ONE packed f32
/// tensor `scales (N+M,)` = `cat(col_scale, row_scale)`: `s_col = scales[0..N]`,
/// `s_row = scales[N..N+M]` (the kernel reads `s_row = s_col_ptr + N`).
///
/// Order matters for alignment: the epilogue's `RowBroadcast` load of `s_col`
/// is vectorized (8 f32 / 32 B), so its base must be 32-B aligned — putting
/// `col_scale` first keeps it at the buffer base. `s_row` then sits at offset N,
/// which is 32-B aligned because N (out_features) is always a multiple of 8
/// (the bf16 store alignment requires it anyway).
struct Int8GemmDequant(EpilogueOut);
impl candle_core::CustomOp3 for Int8GemmDequant {
    fn name(&self) -> &'static str {
        match self.0 {
            EpilogueOut::Bf16 => "int8-gemm-dequant-bf16",
            EpilogueOut::F16 => "int8-gemm-dequant-f16",
        }
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
        candle_core::bail!("int8-gemm-dequant is CUDA-only")
    }
    fn cuda_fwd(
        &self,
        a: &CudaStorage,
        a_l: &Layout,
        b: &CudaStorage,
        b_l: &Layout,
        s: &CudaStorage,
        s_l: &Layout,
    ) -> candle_core::Result<(CudaStorage, Shape)> {
        let dev = a.device().clone();
        let (m, k) = a_l.shape().dims2()?;
        let (n, k2) = b_l.shape().dims2()?;
        if k != k2 {
            candle_core::bail!("int8-gemm-dequant K mismatch: {k} vs {k2}");
        }
        let packed = s_l.shape().dims1()?;
        if packed != m + n {
            candle_core::bail!(
                "int8-gemm-dequant packed scales len {packed} != N+M = {}",
                n + m
            );
        }
        // U8 storage holding int8 bytes; scales are f32 (col ++ row).
        let ao = dense_byte_offset::<u8>(a_l, "int8-gemm-dequant a")?;
        let bo = dense_byte_offset::<u8>(b_l, "int8-gemm-dequant b")?;
        let so = dense_byte_offset::<f32>(s_l, "int8-gemm-dequant scales")?;
        let a = a.as_cuda_slice::<u8>()?;
        let b = b.as_cuda_slice::<u8>()?;
        let s = s.as_cuda_slice::<f32>()?;
        let stream = dev.cuda_stream();
        let (ap, _ga) = a.device_ptr(&stream);
        let (bp, _gb) = b.device_ptr(&stream);
        let (sp, _gs) = s.device_ptr(&stream);
        let (ap, bp, sp) = (ap + ao, bp + bo, sp + so);
        // CUTLASS 128-bit operand loads need 16-B aligned A/B; the vectorized
        // RowBroadcast of s_col needs a 32-B aligned base (see HANDOVER).
        if !ap.is_multiple_of(16) || !bp.is_multiple_of(16) || !sp.is_multiple_of(32) {
            candle_core::bail!(
                "int8-gemm-dequant: misaligned view (a {ap:#x}, b {bp:#x}, scales {sp:#x})"
            );
        }
        let s_col = sp as *const f32;
        // s_row follows s_col in the packed buffer (offset N floats, aligned).
        let s_row = unsafe { s_col.add(n) };
        let (a_i8, b_i8) = (ap as *const i8, bp as *const i8);
        let cu = stream.cu_stream() as *mut std::ffi::c_void;
        let (mi, ni, ki) = (m as i32, n as i32, k as i32);
        // Both variants share every argument but the store type (and launcher).
        let storage = match self.0 {
            EpilogueOut::Bf16 => {
                let out = unsafe { dev.alloc::<half::bf16>(m * n)? };
                {
                    let (op, _go) = out.device_ptr(&stream);
                    let d = op as *mut std::ffi::c_void;
                    let rc = unsafe {
                        int8_gemm_dequant_bf16(d, a_i8, b_i8, s_row, s_col, mi, ni, ki, cu)
                    };
                    if rc != 0 {
                        candle_core::bail!("int8_gemm_dequant_bf16 failed, rc={rc}");
                    }
                }
                CudaStorage::wrap_cuda_slice(out, dev.clone())
            }
            EpilogueOut::F16 => {
                let out = unsafe { dev.alloc::<half::f16>(m * n)? };
                {
                    let (op, _go) = out.device_ptr(&stream);
                    let d = op as *mut std::ffi::c_void;
                    let rc = unsafe {
                        int8_gemm_dequant_f16(d, a_i8, b_i8, s_row, s_col, mi, ni, ki, cu)
                    };
                    if rc != 0 {
                        candle_core::bail!("int8_gemm_dequant_f16 failed, rc={rc}");
                    }
                }
                CudaStorage::wrap_cuda_slice(out, dev.clone())
            }
        };
        Ok((storage, (m, n).into()))
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
        Ok(Self {
            w_i8,
            col_scale,
            hadamard: r.clone(),
        })
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
        self.forward_as(x, EpilogueOut::Bf16)
    }

    /// Forward on `x (..., K)` bf16 -> `(..., N)` in the epilogue's `out` type
    /// (f16 feeds SageAttention's FP16 P·V without a separate cast kernel).
    pub fn forward_as(&self, x: &Tensor, out_ty: EpilogueOut) -> Result<Tensor> {
        let dims = x.dims().to_vec();
        let k = *dims.last().unwrap();
        let m: usize = dims[..dims.len() - 1].iter().product();
        let x2 = x.reshape((m, k))?;
        let xr = crate::model::rotation::rotate(&x2, &self.hadamard)?; // (M,K) bf16
        let (x_i8, row_scale) = quantize_rows_fused(&xr)?; // fused amax+quantize
        let n = self.col_scale.dim(0)?;
        // Pack the two scale vectors into one tensor (candle has no CustomOp4):
        // scales[0..N] = s_col, scales[N..N+M] = s_row. col first so the
        // vectorized RowBroadcast load of s_col starts at the aligned base.
        let scales = Tensor::cat(&[&self.col_scale, &row_scale], 0)?; // (N+M,) f32

        // Fused INT8 GEMM + per-row/per-col dequant epilogue -> (M,N) out_ty.
        let out = x_i8.apply_op3(&self.w_i8, &scales, Int8GemmDequant(out_ty))?;
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
    let y_ref = x
        .to_dtype(DType::F32)?
        .matmul(&w.to_dtype(DType::F32)?.t()?)?;
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
    let a_u8 = Tensor::from_vec(
        ai.iter().map(|&v| v as u8).collect::<Vec<u8>>(),
        (m, k),
        &dev,
    )?;
    let b_u8 = Tensor::from_vec(
        bi.iter().map(|&v| v as u8).collect::<Vec<u8>>(),
        (n, k),
        &dev,
    )?;
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

/// Bit-exactness of the fused dequant epilogue, both store types. Raw int8
/// operands + positive f32 scales go through `Int8GemmDequant` (bf16 and f16),
/// and a host reference applies the epilogue's exact op order —
/// `(acc as f32 * s_row[m]) * s_col[n]`, then one round-to-nearest-even to the
/// 16-bit type. M is odd (37) to exercise the packed-scale alignment that once
/// faulted at M=4117. Returns the mismatch counts `(bf16, f16)` out of M·N.
pub fn self_test_epilogue() -> Result<(usize, usize)> {
    use candle_core::Device;
    let dev = Device::new_cuda(0)?;
    let (m, n, k) = (37usize, 256usize, 512usize);
    let ai: Vec<i8> = (0..m * k).map(|i| ((i * 7) % 255) as i8).collect();
    let bi: Vec<i8> = (0..n * k).map(|i| ((i * 13 + 5) % 255) as i8).collect();
    // Scales spanning several binades so both 16-bit roundings are exercised.
    let s_row: Vec<f32> = (0..m).map(|i| 1e-4 * (1.0 + i as f32 * 0.37)).collect();
    let s_col: Vec<f32> = (0..n).map(|j| 1e-3 * (0.5 + j as f32 * 0.011)).collect();
    let to_u8 = |v: &[i8]| v.iter().map(|&x| x as u8).collect::<Vec<u8>>();
    let a = Tensor::from_vec(to_u8(&ai), (m, k), &dev)?;
    let b = Tensor::from_vec(to_u8(&bi), (n, k), &dev)?;
    let mut packed = s_col.clone();
    packed.extend_from_slice(&s_row);
    let scales = Tensor::from_vec(packed, n + m, &dev)?;
    let bf = a.apply_op3(&b, &scales, Int8GemmDequant(EpilogueOut::Bf16))?;
    let hf = a.apply_op3(&b, &scales, Int8GemmDequant(EpilogueOut::F16))?;
    let bf: Vec<half::bf16> = bf.flatten_all()?.to_vec1()?;
    let hf: Vec<half::f16> = hf.flatten_all()?.to_vec1()?;
    let (mut bad_bf, mut bad_hf) = (0usize, 0usize);
    for mi in 0..m {
        for ni in 0..n {
            let acc: i32 = (0..k)
                .map(|kk| ai[mi * k + kk] as i32 * bi[ni * k + kk] as i32)
                .sum();
            let v = (acc as f32 * s_row[mi]) * s_col[ni];
            let i = mi * n + ni;
            if half::bf16::from_f32(v).to_bits() != bf[i].to_bits() {
                bad_bf += 1;
            }
            if half::f16::from_f32(v).to_bits() != hf[i].to_bits() {
                bad_hf += 1;
            }
        }
    }
    Ok((bad_bf, bad_hf))
}

/// The f16 epilogue at the DiT to_v shape (M=4117, K=N=4096) vs the bf16
/// epilogue + a bf16->f16 cast (the path it replaces). Single vs double
/// rounding differ by at most 1 bf16 ulp, so this returns `(max_abs_diff,
/// max_abs_ref)` in f32 for a relative bound check.
pub fn self_test_f16_vs_cast() -> Result<(f32, f32)> {
    use candle_core::{DType, Device};
    let dev = Device::new_cuda(0)?;
    let (m, n, k) = (4117usize, 4096usize, 4096usize);
    let w = (Tensor::randn(0f32, 1f32, (n, k), &dev)? * 0.02)?.to_dtype(DType::BF16)?;
    let x = Tensor::randn(0f32, 1f32, (m, k), &dev)?.to_dtype(DType::BF16)?;
    let r = crate::model::rotation::regular_hadamard_256(&dev)?;
    let cr = ConvRotLinear::from_weight(&w, &r)?;
    let y16 = cr.forward_as(&x, EpilogueOut::F16)?.to_dtype(DType::F32)?;
    let yref = cr.forward(&x)?.to_dtype(DType::F16)?.to_dtype(DType::F32)?;
    let d = (&y16 - &yref)?.abs()?.max_all()?.to_scalar::<f32>()?;
    let a = yref.abs()?.max_all()?.to_scalar::<f32>()?;
    Ok((d, a))
}

/// Offset-view regression (`qwen-image-rs-b1-off-prompt`): every convrot
/// bridge runs on dense views with a NONZERO storage offset (row narrows of a
/// larger tensor, which candle keeps zero-copy) and on fresh offset-0 copies;
/// the outputs must be bit-identical. Returns `(op, bit_identical)` per op.
pub fn self_test_offset_views() -> Result<Vec<(&'static str, bool)>> {
    use candle_core::{DType, Device};
    let dev = Device::new_cuda(0)?;
    let (drop, m, n, k) = (5usize, 37usize, 256usize, 512usize);
    let rows = |r: usize, cols: usize, seed: usize| -> Result<Tensor> {
        let v: Vec<u8> = (0..(drop + r) * cols)
            .map(|i| (((i * 7 + seed) % 255) as i8) as u8)
            .collect();
        Ok(Tensor::from_vec(v, (drop + r, cols), &dev)?.narrow(0, drop, r)?)
    };
    let (a, b) = (rows(m, k, 1)?, rows(n, k, 3)?);
    anyhow::ensure!(
        a.layout().start_offset() == drop * k,
        "test premise: the narrow must be a zero-copy offset view"
    );
    let (af, bf) = (a.force_contiguous()?, b.force_contiguous()?);
    let u8s = |t: &Tensor| -> Result<Vec<u8>> { Ok(t.flatten_all()?.to_vec1::<u8>()?) };

    // int8 GEMM (i32 out).
    let gemm_ok = int8_gemm(&a, &b)?.flatten_all()?.to_vec1::<i32>()?
        == int8_gemm(&af, &bf)?.flatten_all()?.to_vec1::<i32>()?;

    // Fused dequant epilogue, with the packed scales an offset view too.
    // 8 f32 = 32 B keeps the s_col base at the epilogue's 32-B alignment.
    let sdrop = 8usize;
    let sv: Vec<f32> = (0..sdrop + n + m)
        .map(|i| 1e-3 * (1.0 + i as f32 * 0.01))
        .collect();
    let sc = Tensor::from_vec(sv, sdrop + n + m, &dev)?.narrow(0, sdrop, n + m)?;
    let scf = sc.force_contiguous()?;
    let deq = |a: &Tensor, b: &Tensor, s: &Tensor| -> Result<Vec<u16>> {
        Ok(a.apply_op3(b, s, Int8GemmDequant(EpilogueOut::Bf16))?
            .flatten_all()?
            .to_vec1::<half::bf16>()?
            .iter()
            .map(|v| v.to_bits())
            .collect())
    };
    let deq_ok = deq(&a, &b, &sc)? == deq(&af, &bf, &scf)?;

    // Activation quantizers on an offset bf16 view.
    let x = Tensor::randn(0f32, 1f32, (drop + m, k), &dev)?
        .to_dtype(DType::BF16)?
        .narrow(0, drop, m)?;
    let xf = x.force_contiguous()?;
    let (q, rs) = quantize_rows_fused(&x)?;
    let (qf, rsf) = quantize_rows_fused(&xf)?;
    let fused_ok = u8s(&q)? == u8s(&qf)? && rs.to_vec1::<f32>()? == rsf.to_vec1::<f32>()?;
    let inv = Tensor::from_vec(vec![100f32; drop + m], drop + m, &dev)?.narrow(0, drop, m)?;
    let invf = inv.force_contiguous()?;
    let rows_ok =
        u8s(&x.apply_op2(&inv, QuantizeRows)?)? == u8s(&xf.apply_op2(&invf, QuantizeRows)?)?;
    Ok(vec![
        ("int8_gemm", gemm_ok),
        ("int8_gemm_dequant", deq_ok),
        ("quantize_rows_fused", fused_ok),
        ("quantize_rows", rows_ok),
    ])
}
