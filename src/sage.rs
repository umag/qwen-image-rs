//! SageAttention INT8-QK / FP16-PV bridge. A candle `CustomOp3` that, given
//! `q,k` (bf16) and `v` (fp16) in layout `(B, H, S, D)`, quantizes q,k to INT8
//! per-block, runs the vendored fused attention kernel (kernels/sage/), and
//! returns bf16 output. Only compiled under the `sage` feature (implies `cuda`).
//!
//! INT8 rides in U8 storage (candle has no I8), same as the convrot bridge.

use candle_core::backend::BackendStorage;
use candle_core::cuda_backend::cudarc::driver::DevicePtr;
use candle_core::{CpuStorage, CudaStorage, DType, Layout, Shape, Tensor};
use std::ffi::c_void;

use crate::Result;

extern "C" {
    fn sage_quant_q(
        inp: *const c_void,
        out_i8: *mut c_void,
        scale: *mut f32,
        b: i32,
        h: i32,
        n: i32,
        stream: *mut c_void,
    );
    fn sage_quant_k(
        inp: *const c_void,
        out_i8: *mut c_void,
        scale: *mut f32,
        b: i32,
        h: i32,
        n: i32,
        stream: *mut c_void,
    );
    #[allow(clippy::too_many_arguments)]
    fn sage_attn(
        q_i8: *const c_void,
        k_i8: *const c_void,
        v_f16: *const c_void,
        o_bf16: *mut c_void,
        q_scale: *const f32,
        k_scale: *const f32,
        b: i32,
        hq: i32,
        hk: i32,
        qo: i32,
        kv: i32,
        sm_scale: f32,
        is_causal: i32,
        stream: *mut c_void,
    );
}

extern "C" {
    fn sage_quant_q_bshd(
        inp: *const c_void,
        out_i8: *mut c_void,
        scale: *mut f32,
        b: i32,
        h: i32,
        n: i32,
        sbz_in: u32,
        sseq_in: u32,
        sh_in: u32,
        stream: *mut c_void,
    );
    fn sage_quant_k_bshd(
        inp: *const c_void,
        out_i8: *mut c_void,
        scale: *mut f32,
        b: i32,
        h: i32,
        n: i32,
        sbz_in: u32,
        sseq_in: u32,
        sh_in: u32,
        stream: *mut c_void,
    );
    #[allow(clippy::too_many_arguments)]
    fn sage_attn_bshd(
        q_i8: *const c_void,
        k_i8: *const c_void,
        v_f16: *const c_void,
        o_bf16: *mut c_void,
        q_scale: *const f32,
        k_scale: *const f32,
        b: i32,
        hq: i32,
        hk: i32,
        qo: i32,
        kv: i32,
        sbz_v: u32,
        sseq_v: u32,
        sh_v: u32,
        sbz_o: u32,
        sseq_o: u32,
        sh_o: u32,
        sm_scale: f32,
        is_causal: i32,
        stream: *mut c_void,
    );
}

struct SageAttn {
    scale: f32,
    causal: bool,
}

impl candle_core::CustomOp3 for SageAttn {
    fn name(&self) -> &'static str {
        "sage-attn"
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
        candle_core::bail!("sage-attn is CUDA-only")
    }

    fn cuda_fwd(
        &self,
        q: &CudaStorage,
        ql: &Layout,
        k: &CudaStorage,
        kl: &Layout,
        v: &CudaStorage,
        _vl: &Layout,
    ) -> candle_core::Result<(CudaStorage, Shape)> {
        let dev = q.device().clone();
        let (b, hq, s, d) = ql.shape().dims4()?;
        let (_, hk, skv, _) = kl.shape().dims4()?;
        let q_s = q.as_cuda_slice::<half::bf16>()?;
        let k_s = k.as_cuda_slice::<half::bf16>()?;
        let v_s = v.as_cuda_slice::<half::f16>()?;
        let stream = dev.cuda_stream();

        let nq_blk = s.div_ceil(128);
        let nk_blk = skv.div_ceil(64);
        let q_i8 = unsafe { dev.alloc::<u8>(b * hq * s * d)? };
        let k_i8 = unsafe { dev.alloc::<u8>(b * hk * skv * d)? };
        let q_scale = unsafe { dev.alloc::<f32>(b * hq * nq_blk)? };
        let k_scale = unsafe { dev.alloc::<f32>(b * hk * nk_blk)? };
        let o = unsafe { dev.alloc::<half::bf16>(b * hq * s * d)? };

        {
            let (qp, _a) = q_s.device_ptr(&stream);
            let (kp, _b) = k_s.device_ptr(&stream);
            let (vp, _c) = v_s.device_ptr(&stream);
            let (qip, _d) = q_i8.device_ptr(&stream);
            let (kip, _e) = k_i8.device_ptr(&stream);
            let (qsp, _f) = q_scale.device_ptr(&stream);
            let (ksp, _g) = k_scale.device_ptr(&stream);
            let (op, _h) = o.device_ptr(&stream);
            let st = stream.cu_stream() as *mut c_void;
            unsafe {
                sage_quant_q(
                    qp as *const c_void,
                    qip as *mut c_void,
                    qsp as *mut f32,
                    b as i32,
                    hq as i32,
                    s as i32,
                    st,
                );
                sage_quant_k(
                    kp as *const c_void,
                    kip as *mut c_void,
                    ksp as *mut f32,
                    b as i32,
                    hk as i32,
                    skv as i32,
                    st,
                );
                sage_attn(
                    qip as *const c_void,
                    kip as *const c_void,
                    vp as *const c_void,
                    op as *mut c_void,
                    qsp as *const f32,
                    ksp as *const f32,
                    b as i32,
                    hq as i32,
                    hk as i32,
                    s as i32,
                    skv as i32,
                    self.scale,
                    self.causal as i32,
                    st,
                );
            }
        }
        Ok((CudaStorage::wrap_cuda_slice(o, dev), (b, hq, s, d).into()))
    }
}

/// INT8-QK / FP16-PV attention. `q,k,v` are `(B, H, S, D)` bf16 on CUDA;
/// `scale` is the softmax scale (1/sqrt(D)); `causal` selects the causal mask.
/// Returns `(B, H, S, D)` bf16. D must be a multiple of 64.
pub fn sage_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f32,
    causal: bool,
) -> Result<Tensor> {
    let q = q.contiguous()?;
    let k = k.contiguous()?;
    let vf = v.to_dtype(DType::F16)?.contiguous()?;
    Ok(q.apply_op3(&k, &vf, SageAttn { scale, causal })?)
}

extern "C" {
    #[allow(clippy::too_many_arguments)]
    fn sage_rope_quant_bshd_launch(
        inp: *const c_void,
        cos: *const c_void,
        sin: *const c_void,
        out_i8: *mut c_void,
        scale: *mut f32,
        b: i32,
        h: i32,
        n: i32,
        sbz_in: u32,
        sseq_in: u32,
        sh_in: u32,
        sseq_cs: u32,
        is_q: i32,
        stream: *mut c_void,
    );
}

/// Head dim the vendored kernels are instantiated for (`HEAD_DIM` in sage_ffi.cu).
const SAGE_HEAD_DIM: usize = 128;

/// Which attention operand a quantized tensor is: it fixes the per-block INT8
/// granularity the attention kernel indexes (`CTA_Q` = 128 query tokens per
/// scale, `CTA_K` = 64 key tokens per scale).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QkRole {
    Query,
    Key,
}

impl QkRole {
    fn block(self) -> usize {
        match self {
            QkRole::Query => 128,
            QkRole::Key => 64,
        }
    }
}

/// Value object: one attention operand already INT8-quantized for the sage
/// kernel. `packed` is a U8 tensor holding the int8 values in HND-contiguous
/// order `(b, h, n, 128)` followed by the per-block f32 scales `(b, h,
/// ceil(n/block))` — the exact layout the attention kernel's per-block
/// `q_scale_idx` / `k_scale_idx` expect, with block 0 at the operand's first
/// token. Produced by [`rope_quant_bshd`] (fused) or [`quant_bshd`] (unfused).
pub struct QuantizedQk {
    packed: Tensor,
    b: usize,
    h: usize,
    n: usize,
    role: QkRole,
}

impl QuantizedQk {
    fn int8_bytes(b: usize, h: usize, n: usize) -> usize {
        b * h * n * SAGE_HEAD_DIM
    }
    fn packed_bytes(b: usize, h: usize, n: usize, role: QkRole) -> usize {
        Self::int8_bytes(b, h, n) + 4 * b * h * n.div_ceil(role.block())
    }
    /// Raw packed bytes (int8 region then f32 scales) — for bit-exact tests.
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        Ok(self.packed.to_vec1::<u8>()?)
    }
    /// Byte length of the int8 region (the scales follow it).
    pub fn int8_len(&self) -> usize {
        Self::int8_bytes(self.b, self.h, self.n)
    }
}

/// Validate a bf16 `(B, S, H, D)` operand view for the quant kernels: D=128,
/// head_dim stride-1, and the 16-byte alignment the kernels' float4 loads need.
fn check_qk_view(xl: &Layout, what: &str) -> candle_core::Result<(usize, usize, usize)> {
    let (b, n, h, d) = xl.shape().dims4()?;
    let st = xl.stride();
    if d != SAGE_HEAD_DIM || st[3] != 1 {
        candle_core::bail!(
            "{what}: need head_dim {SAGE_HEAD_DIM} stride-1, got d={d} stride {st:?}"
        );
    }
    // float4 = 8 bf16: base offset and every outer stride must be multiples of 8.
    if !xl.start_offset().is_multiple_of(8) || st[..3].iter().any(|s| !s.is_multiple_of(8)) {
        candle_core::bail!(
            "{what}: view not 16-byte aligned (offset {}, strides {st:?})",
            xl.start_offset()
        );
    }
    Ok((b, n, h))
}

/// Unfused per-block INT8 quantization of a bf16 `(B,S,H,D)` view (the
/// equivalence oracle for the fused op, and the quant step of
/// [`sage_attention_bshd`]).
struct QuantBshd {
    role: QkRole,
}

impl candle_core::CustomOp1 for QuantBshd {
    fn name(&self) -> &'static str {
        "sage-quant-bshd"
    }

    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> candle_core::Result<(CpuStorage, Shape)> {
        candle_core::bail!("sage-quant-bshd is CUDA-only")
    }

    fn cuda_fwd(&self, x: &CudaStorage, xl: &Layout) -> candle_core::Result<(CudaStorage, Shape)> {
        let dev = x.device().clone();
        let (b, n, h) = check_qk_view(xl, "sage-quant-bshd")?;
        let st = xl.stride();
        let x_s = x.as_cuda_slice::<half::bf16>()?;
        let stream = dev.cuda_stream();
        let total = QuantizedQk::packed_bytes(b, h, n, self.role);
        let i8_len = QuantizedQk::int8_bytes(b, h, n);
        let out = unsafe { dev.alloc::<u8>(total)? };
        {
            let (xp, _a) = x_s.device_ptr(&stream);
            let (op, _b) = out.device_ptr(&stream);
            let xp = xp as usize + xl.start_offset() * std::mem::size_of::<half::bf16>();
            let sp = (op as usize + i8_len) as *mut f32;
            let st_ = stream.cu_stream() as *mut c_void;
            let f = match self.role {
                QkRole::Query => sage_quant_q_bshd,
                QkRole::Key => sage_quant_k_bshd,
            };
            unsafe {
                f(
                    xp as *const c_void,
                    op as *mut c_void,
                    sp,
                    b as i32,
                    h as i32,
                    n as i32,
                    st[0] as u32,
                    st[1] as u32,
                    st[2] as u32,
                    st_,
                );
            }
        }
        Ok((CudaStorage::wrap_cuda_slice(out, dev), total.into()))
    }
}

/// Fused interleaved-RoPE + per-block INT8 quantization of a PRE-rope bf16
/// `(B,S,H,D)` view; cos/sin are the matching `(S, D/2)` rows (row 0 = the
/// view's first token).
struct RopeQuantBshd {
    role: QkRole,
}

impl candle_core::CustomOp3 for RopeQuantBshd {
    fn name(&self) -> &'static str {
        "sage-rope-quant-bshd"
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
        candle_core::bail!("sage-rope-quant-bshd is CUDA-only")
    }

    fn cuda_fwd(
        &self,
        x: &CudaStorage,
        xl: &Layout,
        cos: &CudaStorage,
        cl: &Layout,
        sin: &CudaStorage,
        sl: &Layout,
    ) -> candle_core::Result<(CudaStorage, Shape)> {
        let dev = x.device().clone();
        let (b, n, h) = check_qk_view(xl, "sage-rope-quant-bshd")?;
        let (cn, dhalf) = cl.shape().dims2()?;
        if cn != n || dhalf * 2 != SAGE_HEAD_DIM || sl.shape() != cl.shape() {
            candle_core::bail!(
                "sage-rope-quant-bshd: cos {:?} / sin {:?} vs x ({n} tokens, d {SAGE_HEAD_DIM})",
                cl.shape(),
                sl.shape()
            );
        }
        // Rows of D/2 contiguous bf16 (a dim-0 narrow of (S, D/2) keeps this).
        if !cl.is_contiguous() || !sl.is_contiguous() {
            candle_core::bail!("sage-rope-quant-bshd: cos/sin must be row-contiguous (S, D/2)");
        }
        let st = xl.stride();
        let x_s = x.as_cuda_slice::<half::bf16>()?;
        let cos_s = cos.as_cuda_slice::<half::bf16>()?;
        let sin_s = sin.as_cuda_slice::<half::bf16>()?;
        let stream = dev.cuda_stream();
        let total = QuantizedQk::packed_bytes(b, h, n, self.role);
        let i8_len = QuantizedQk::int8_bytes(b, h, n);
        let out = unsafe { dev.alloc::<u8>(total)? };
        {
            let bf = std::mem::size_of::<half::bf16>();
            let (xp, _a) = x_s.device_ptr(&stream);
            let (cp, _b) = cos_s.device_ptr(&stream);
            let (sp, _c) = sin_s.device_ptr(&stream);
            let (op, _d) = out.device_ptr(&stream);
            let xp = xp as usize + xl.start_offset() * bf;
            let cp = cp as usize + cl.start_offset() * bf;
            let sp = sp as usize + sl.start_offset() * bf;
            let scp = (op as usize + i8_len) as *mut f32;
            unsafe {
                sage_rope_quant_bshd_launch(
                    xp as *const c_void,
                    cp as *const c_void,
                    sp as *const c_void,
                    op as *mut c_void,
                    scp,
                    b as i32,
                    h as i32,
                    n as i32,
                    st[0] as u32,
                    st[1] as u32,
                    st[2] as u32,
                    dhalf as u32,
                    (self.role == QkRole::Query) as i32,
                    stream.cu_stream() as *mut c_void,
                );
            }
        }
        Ok((CudaStorage::wrap_cuda_slice(out, dev), total.into()))
    }
}

/// Fused RoPE + INT8 quant: rotate the PRE-rope bf16 view `x (B,S,H,D)` with
/// `cos,sin (S, D/2)` (the rows for exactly the view's tokens — narrow them on
/// dim 0 alongside `x`'s dim-1 narrow) and quantize per `role` block. Equals
/// `quant_bshd(rope_i_bshd(x, cos, sin), role)` bit-for-bit.
pub fn rope_quant_bshd(
    x: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    role: QkRole,
) -> Result<QuantizedQk> {
    let (b, n, h, _) = x.dims4()?;
    let packed = x.apply_op3(cos, sin, RopeQuantBshd { role })?;
    Ok(QuantizedQk {
        packed,
        b,
        h,
        n,
        role,
    })
}

/// Unfused per-block INT8 quant of an (already rotated) bf16 `(B,S,H,D)` view.
pub fn quant_bshd(x: &Tensor, role: QkRole) -> Result<QuantizedQk> {
    let (b, n, h, _) = x.dims4()?;
    let packed = x.apply_op1(QuantBshd { role })?;
    Ok(QuantizedQk {
        packed,
        b,
        h,
        n,
        role,
    })
}

/// Attention over pre-quantized q/k: the inputs are the q/k packed buffers and
/// the f16 `(B,S,H,D)` V view; dims come from the [`QuantizedQk`]s.
struct SageAttnPrequant {
    scale: f32,
    causal: bool,
    b: usize,
    hq: usize,
    sq: usize,
    hk: usize,
    skv: usize,
}

impl candle_core::CustomOp3 for SageAttnPrequant {
    fn name(&self) -> &'static str {
        "sage-attn-prequant"
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
        candle_core::bail!("sage-attn-prequant is CUDA-only")
    }

    fn cuda_fwd(
        &self,
        q: &CudaStorage,
        ql: &Layout,
        k: &CudaStorage,
        kl: &Layout,
        v: &CudaStorage,
        vl: &Layout,
    ) -> candle_core::Result<(CudaStorage, Shape)> {
        let dev = q.device().clone();
        let (b, hq, sq, hk, skv) = (self.b, self.hq, self.sq, self.hk, self.skv);
        let d = SAGE_HEAD_DIM;
        let q_len = QuantizedQk::packed_bytes(b, hq, sq, QkRole::Query);
        let k_len = QuantizedQk::packed_bytes(b, hk, skv, QkRole::Key);
        if ql.start_offset() != 0
            || ql.shape().elem_count() != q_len
            || kl.start_offset() != 0
            || kl.shape().elem_count() != k_len
        {
            candle_core::bail!("sage-attn-prequant: packed q/k size mismatch vs dims");
        }
        let (vb, vs, vh, vd) = vl.shape().dims4()?;
        let vst = vl.stride();
        if vb != b || vs != skv || vh != hk || vd != d || vst[3] != 1 {
            candle_core::bail!(
                "sage-attn-prequant: v {:?} vs k ({b},{skv},{hk},{d})",
                vl.shape()
            );
        }
        let q_s = q.as_cuda_slice::<u8>()?;
        let k_s = k.as_cuda_slice::<u8>()?;
        let v_s = v.as_cuda_slice::<half::f16>()?;
        let stream = dev.cuda_stream();
        let o = unsafe { dev.alloc::<half::bf16>(b * sq * hq * d)? };
        // Output is a fresh contiguous (B, sq, Hq, D) buffer: BSHD strides.
        let sbz_o = (sq * hq * d) as u32;
        let sseq_o = (hq * d) as u32;
        let sh_o = d as u32;
        {
            let (qp, _a) = q_s.device_ptr(&stream);
            let (kp, _b) = k_s.device_ptr(&stream);
            let (vp, _c) = v_s.device_ptr(&stream);
            let (op, _d) = o.device_ptr(&stream);
            let qsp = (qp as usize + QuantizedQk::int8_bytes(b, hq, sq)) as *const f32;
            let ksp = (kp as usize + QuantizedQk::int8_bytes(b, hk, skv)) as *const f32;
            let vp = vp as usize + vl.start_offset() * std::mem::size_of::<half::f16>();
            unsafe {
                sage_attn_bshd(
                    qp as *const c_void,
                    kp as *const c_void,
                    vp as *const c_void,
                    op as *mut c_void,
                    qsp,
                    ksp,
                    b as i32,
                    hq as i32,
                    hk as i32,
                    sq as i32,
                    skv as i32,
                    vst[0] as u32,
                    vst[1] as u32,
                    vst[2] as u32,
                    sbz_o,
                    sseq_o,
                    sh_o,
                    self.scale,
                    self.causal as i32,
                    stream.cu_stream() as *mut c_void,
                );
            }
        }
        Ok((CudaStorage::wrap_cuda_slice(o, dev), (b, sq, hq, d).into()))
    }
}

/// INT8-QK / FP16-PV attention over pre-quantized `q` ([`QkRole::Query`]) and
/// `k` ([`QkRole::Key`]); `v` is the f16 `(B,S,H,D)` view (may be an S-axis
/// narrow). Returns `(B, q.n, H, D)` bf16.
pub fn sage_attention_quantized(
    q: &QuantizedQk,
    k: &QuantizedQk,
    v: &Tensor,
    scale: f32,
    causal: bool,
) -> Result<Tensor> {
    if q.role != QkRole::Query || k.role != QkRole::Key || q.b != k.b {
        anyhow::bail!("sage_attention_quantized: need (Query, Key) operands of equal batch");
    }
    let op = SageAttnPrequant {
        scale,
        causal,
        b: q.b,
        hq: q.h,
        sq: q.n,
        hk: k.h,
        skv: k.n,
    };
    Ok(q.packed.apply_op3(&k.packed, v, op)?)
}

/// INT8-QK / FP16-PV attention, BSHD-native. `q,k` are `(B,S,H,D)` bf16, `v` is
/// `(B,S,H,D)` f16 (all may be S-axis narrows — start_offset + strides are
/// honored, so the block-causal narrows are zero-copy). `scale` is 1/sqrt(D);
/// `causal` selects the causal mask. Returns `(B,S,H,D)` bf16. D must be 128.
/// = unfused quant + [`sage_attention_quantized`]; the DiT uses the fused
/// [`rope_quant_bshd`] instead, this stays as the equivalence oracle.
pub fn sage_attention_bshd(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f32,
    causal: bool,
) -> Result<Tensor> {
    let q8 = quant_bshd(q, QkRole::Query)?;
    let k8 = quant_bshd(k, QkRole::Key)?;
    sage_attention_quantized(&q8, &k8, v, scale, causal)
}

/// Validate INT8 SageAttention vs an f32 softmax-attention reference: cosine.
pub fn self_test(causal: bool) -> Result<f32> {
    use candle_core::{Device, D};
    let dev = Device::new_cuda(0)?;
    let (b, h, s, d) = (1usize, 4usize, 256usize, 128usize);
    let scale = 1.0 / (d as f64).sqrt();
    let q = Tensor::randn(0f32, 1f32, (b, h, s, d), &dev)?.to_dtype(DType::BF16)?;
    let k = Tensor::randn(0f32, 1f32, (b, h, s, d), &dev)?.to_dtype(DType::BF16)?;
    let v = Tensor::randn(0f32, 1f32, (b, h, s, d), &dev)?.to_dtype(DType::BF16)?;

    let y = sage_attention(&q, &k, &v, scale as f32, causal)?.to_dtype(DType::F32)?;

    // f32 reference.
    let qf = q.to_dtype(DType::F32)?;
    let kf = k.to_dtype(DType::F32)?;
    let vf = v.to_dtype(DType::F32)?;
    let mut scores = (qf.matmul(&kf.transpose(D::Minus1, D::Minus2)?)? * scale)?;
    if causal {
        let mut mask = vec![0f32; s * s];
        for i in 0..s {
            for j in (i + 1)..s {
                mask[i * s + j] = f32::NEG_INFINITY;
            }
        }
        let m = Tensor::from_vec(mask, (1, 1, s, s), &dev)?;
        scores = scores.broadcast_add(&m)?;
    }
    let attn = candle_nn::ops::softmax_last_dim(&scores)?;
    let yref = attn.matmul(&vf)?;

    let dot = (&y * &yref)?.sum_all()?.to_scalar::<f32>()?;
    let na = y.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt();
    let nb = yref.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt();
    Ok(dot / (na * nb + 1e-8))
}

/// A/B equivalence: the BSHD path must equal the retained BHSD `sage_attention`
/// (same kernel, same INT8 quant, only the read/write layout differs) on the
/// exact block-causal split the DiT uses (causal text prefix + non-causal image
/// queries). Exercises the S-axis narrows' start_offset. Returns (cosine, maxabs).
pub fn self_test_bshd() -> Result<(f32, f32)> {
    use candle_core::{DType, Device};
    let (b, h, s, d, txt) = (1usize, 4usize, 160usize, 128usize, 32usize);
    let dev = Device::new_cuda(0)?;
    dev.set_seed(42)?; // deterministic inputs
    let sc = (1.0 / (d as f64).sqrt()) as f32;
    let q = Tensor::randn(0f32, 1f32, (b, s, h, d), &dev)?.to_dtype(DType::BF16)?;
    let k = Tensor::randn(0f32, 1f32, (b, s, h, d), &dev)?.to_dtype(DType::BF16)?;
    let v = Tensor::randn(0f32, 1f32, (b, s, h, d), &dev)?.to_dtype(DType::BF16)?;
    let vf = v.to_dtype(DType::F16)?;

    // BHSD reference (existing path): transpose to (B,H,S,D), narrow on dim 2.
    let qh = q.transpose(1, 2)?.contiguous()?;
    let kh = k.transpose(1, 2)?.contiguous()?;
    let vh = vf.transpose(1, 2)?.contiguous()?;
    let ot = sage_attention(
        &qh.narrow(2, 0, txt)?,
        &kh.narrow(2, 0, txt)?,
        &vh.narrow(2, 0, txt)?,
        sc,
        true,
    )?;
    let qi = qh.narrow(2, txt, s - txt)?;
    let oi = sage_attention(&qi, &kh, &vh, sc, false)?;
    let refb = Tensor::cat(&[ot, oi], 2)?
        .transpose(1, 2)?
        .contiguous()?
        .to_dtype(DType::F32)?; // (B,S,H,D)

    // BSHD path (new): narrow on dim 1, no transpose.
    let ot = sage_attention_bshd(
        &q.narrow(1, 0, txt)?,
        &k.narrow(1, 0, txt)?,
        &vf.narrow(1, 0, txt)?,
        sc,
        true,
    )?;
    let qi = q.narrow(1, txt, s - txt)?;
    let oi = sage_attention_bshd(&qi, &k, &vf, sc, false)?;
    let ours = Tensor::cat(&[ot, oi], 1)?.to_dtype(DType::F32)?; // (B,S,H,D)

    // NaN-aware host compare (see nan_aware_compare).
    nan_aware_compare(&ours, &refb)
}

/// NaN-aware element-wise comparison of two same-shape tensors on the host:
/// returns (cosine over finite pairs, maxabs over finite pairs). Shared by the
/// equivalence self-tests.
fn nan_aware_compare(ours: &Tensor, refb: &Tensor) -> Result<(f32, f32)> {
    // Element-wise on the host, NaN-aware. The equivalence tests prove two paths
    // feed the same kernel identically — NOT that the vendored INT8 kernel never
    // NaNs. On some synthetic randn draws that kernel
    // emits a NaN row (a per-block-quant edge case; real model activations don't
    // trigger it — dit-forward stays clean). Both paths emit it identically, so
    // a position where BOTH are NaN is a MATCH; a one-sided NaN is a real
    // mismatch. maxabs (over finite pairs) is the bit-exactness signal; cosine
    // is over the finite pairs. A one-sided NaN forces maxabs -> NaN (fails).
    let a = ours.flatten_all()?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
    let bvec = refb.flatten_all()?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
    let (mut dot, mut na, mut nb, mut maxabs) = (0f64, 0f64, 0f64, 0f64);
    let mut mismatched_nans = 0usize;
    for (x, y) in a.iter().zip(bvec.iter()) {
        let (x, y) = (*x as f64, *y as f64);
        match (x.is_finite(), y.is_finite()) {
            (true, true) => {
                dot += x * y;
                na += x * x;
                nb += y * y;
                let diff = (x - y).abs();
                if diff > maxabs {
                    maxabs = diff;
                }
            }
            // both non-finite: a match only if the same class (NaN vs NaN, or
            // equal signed infinities).
            (false, false) if x.is_nan() == y.is_nan() && (x.is_nan() || x == y) => {}
            _ => mismatched_nans += 1,
        }
    }
    let cos = if na > 0.0 && nb > 0.0 {
        (dot / (na.sqrt() * nb.sqrt())) as f32
    } else {
        1.0
    };
    let maxabs = if mismatched_nans > 0 {
        f32::NAN
    } else {
        maxabs as f32
    };
    Ok((cos, maxabs))
}

/// Result of [`self_test_rope_quant`].
pub struct RopeQuantReport {
    /// int8 bytes differing between fused and rope-then-quant (all narrows, B=1,2).
    pub int8_mismatches: usize,
    /// f32 scale bytes differing (same).
    pub scale_mismatches: usize,
    /// fused attention vs rope->sage_attention_bshd, worst over B=1,2: cosine.
    pub attn_cos: f32,
    /// ... and maxabs (NaN on a one-sided NaN).
    pub attn_maxabs: f32,
}

/// Equivalence: the fused [`rope_quant_bshd`] must equal `rope_i_bshd` followed
/// by the unfused [`quant_bshd`] bit-for-bit, on every S-axis narrow the DiT
/// issues (q text prefix, k text prefix, q image queries with their cos/sin
/// rows, full k), at B=1 and B=2, with txt=37 / S=293 so the last Q (128) and
/// K (64) blocks are partial. Then the fused block-causal attention must equal
/// the rope-then-`sage_attention_bshd` path.
pub fn self_test_rope_quant() -> Result<RopeQuantReport> {
    use candle_core::Device;
    let (h, s, d, txt) = (4usize, 293usize, 128usize, 37usize);
    let dev = Device::new_cuda(0)?;
    dev.set_seed(7)?;
    let sc = (1.0 / (d as f64).sqrt()) as f32;
    let cos = Tensor::randn(0f32, 1f32, (s, d / 2), &dev)?.to_dtype(DType::BF16)?;
    let sin = Tensor::randn(0f32, 1f32, (s, d / 2), &dev)?.to_dtype(DType::BF16)?;
    let mut rep = RopeQuantReport {
        int8_mismatches: 0,
        scale_mismatches: 0,
        attn_cos: 1.0,
        attn_maxabs: 0.0,
    };
    for b in [1usize, 2] {
        let q = Tensor::randn(0f32, 1f32, (b, s, h, d), &dev)?.to_dtype(DType::BF16)?;
        let k = Tensor::randn(0f32, 1f32, (b, s, h, d), &dev)?.to_dtype(DType::BF16)?;
        let v = Tensor::randn(0f32, 1f32, (b, s, h, d), &dev)?.to_dtype(DType::F16)?;
        let qr = crate::rope::rope_i_bshd(&q, &cos, &sin)?;
        let kr = crate::rope::rope_i_bshd(&k, &cos, &sin)?;
        let fused = |x: &Tensor, off: usize, len: usize, role: QkRole| -> Result<QuantizedQk> {
            rope_quant_bshd(
                &x.narrow(1, off, len)?,
                &cos.narrow(0, off, len)?,
                &sin.narrow(0, off, len)?,
                role,
            )
        };
        let narrows = [
            (&q, &qr, 0, txt, QkRole::Query),
            (&k, &kr, 0, txt, QkRole::Key),
            (&q, &qr, txt, s - txt, QkRole::Query),
            (&k, &kr, 0, s, QkRole::Key),
        ];
        let mut fq = Vec::new();
        for (x, xr, off, len, role) in narrows {
            let f = fused(x, off, len, role)?;
            let r = quant_bshd(&xr.narrow(1, off, len)?, role)?;
            let (fb, rb) = (f.to_bytes()?, r.to_bytes()?);
            if fb.len() != rb.len() {
                anyhow::bail!("packed length mismatch {} vs {}", fb.len(), rb.len());
            }
            let split = f.int8_len();
            for (i, (x, y)) in fb.iter().zip(rb.iter()).enumerate() {
                if x != y {
                    if i < split {
                        rep.int8_mismatches += 1;
                    } else {
                        rep.scale_mismatches += 1;
                    }
                }
            }
            fq.push(f);
        }
        // Block-causal attention exactly as the DiT's attend_bshd issues it.
        let ot = sage_attention_quantized(&fq[0], &fq[1], &v.narrow(1, 0, txt)?, sc, true)?;
        let oi = sage_attention_quantized(&fq[2], &fq[3], &v, sc, false)?;
        let ours = Tensor::cat(&[ot, oi], 1)?;
        let rt = sage_attention_bshd(
            &qr.narrow(1, 0, txt)?,
            &kr.narrow(1, 0, txt)?,
            &v.narrow(1, 0, txt)?,
            sc,
            true,
        )?;
        let ri = sage_attention_bshd(&qr.narrow(1, txt, s - txt)?, &kr, &v, sc, false)?;
        let refb = Tensor::cat(&[rt, ri], 1)?;
        let (c, m) = nan_aware_compare(&ours, &refb)?;
        rep.attn_cos = rep.attn_cos.min(c);
        rep.attn_maxabs = if m.is_nan() || rep.attn_maxabs.is_nan() {
            f32::NAN
        } else {
            rep.attn_maxabs.max(m)
        };
    }
    Ok(rep)
}
