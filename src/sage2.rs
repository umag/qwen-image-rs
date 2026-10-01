//! SageAttention2 (sm89) bridge: INT8 Q·K with **per-thread** scales and **K
//! smoothing** (the key mean over the call's sequence is subtracted before
//! quantization — softmax is invariant to it), FP8 e4m3 P·V with per-channel V
//! scales. The attention kernel is thu-ml's vendored `qk_int_sv_f8_attn_kernel`;
//! the RoPE + quant + V transpose kernels are ours (`kernels/sage/sage2_ffi.cu`).
//! Only compiled under the `sage2` feature (implies `sage`).
//!
//! The DiT picks the attention at run time via [`attention_impl`]
//! (`QIR_SAGE=1|2|2f32`, default `2`), so SageAttention v1 stays an in-binary
//! fallback and A/B runs need no rebuild.

use candle_core::backend::BackendStorage;
use candle_core::cuda_backend::cudarc::driver::DevicePtr;
use candle_core::{CpuStorage, CudaStorage, DType, Layout, Shape, Tensor};
use std::ffi::c_void;
use std::sync::OnceLock;

use crate::sage::{check_qk_view, check_rope_tables};
use crate::Result;

extern "C" {
    #[allow(clippy::too_many_arguments)]
    fn sage2_rope_quant_launch(
        inp: *const c_void,
        cos: *const c_void,
        sin: *const c_void,
        out_i8: *mut c_void,
        scale: *mut f32,
        partial: *mut f32,
        b: i32,
        h: i32,
        n: i32,
        sbz: u32,
        sseq: u32,
        sh: u32,
        sseq_cs: u32,
        is_k: i32,
        stream: *mut c_void,
    ) -> i32;
    #[allow(clippy::too_many_arguments)]
    fn sage2_quant_v_launch(
        v: *const c_void,
        out_fp8: *mut c_void,
        vscale: *mut f32,
        partial: *mut f32,
        b: i32,
        h: i32,
        n: i32,
        lpad: i32,
        scale_max: f32,
        sbz: u32,
        sseq: u32,
        sh: u32,
        stream: *mut c_void,
    ) -> i32;
    fn sage2_quant_layer_launch(tables: *const S2Tasks, ntables: i32, stream: *mut c_void) -> i32;
    #[allow(clippy::too_many_arguments)]
    fn sage2_attn_launch(
        q_i8: *const c_void,
        k_i8: *const c_void,
        v_fp8: *const c_void,
        o_bf16: *mut c_void,
        q_scale: *const f32,
        k_scale: *const f32,
        v_scale: *const f32,
        b: i32,
        hq: i32,
        hk: i32,
        qo: i32,
        kv: i32,
        lpad: i32,
        sbz_o: u32,
        sseq_o: u32,
        sh_o: u32,
        sm_scale: f32,
        is_causal: i32,
        f16_accum: i32,
        stream: *mut c_void,
    ) -> i32;
}

/// Head dim the kernels are instantiated for.
const D: usize = 128;
/// Tokens per partial-reduction chunk (`CHUNK` in sage2_ffi.cu).
const CHUNK: usize = 256;
/// V tile / padding unit (`CTA_K`).
const V_TILE: usize = 64;

fn check_rc(rc: i32, what: &str) -> candle_core::Result<()> {
    if rc != 0 {
        candle_core::bail!("{what}: CUDA launch failed (cudaError {rc})");
    }
    Ok(())
}

/// P·V accumulation of the FP8 attention kernel. It fixes the V quant range:
/// the fp32+fp16 two-level accumulator needs V scaled to |v| <= 2.25 so the
/// fp16 partial sums cannot overflow; fp32 uses the full e4m3 range (448).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PvAccum {
    /// fp32 + fp16 two-level accumulation (upstream's sm89 default, fastest).
    F16,
    /// fp32 accumulation (slower, more accurate).
    F32,
}

impl PvAccum {
    fn scale_max(self) -> f32 {
        match self {
            PvAccum::F16 => 2.25,
            PvAccum::F32 => 448.0,
        }
    }
}

/// Which attention the DiT runs in a `sage2` build.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AttentionImpl {
    /// SageAttention v1 (INT8-QK per-block / FP16-PV) — the fallback.
    Sage1,
    /// SageAttention2 (INT8-QK per-thread + K smoothing / FP8-PV).
    Sage2(PvAccum),
}

/// Parse a `QIR_SAGE` value (`None` = unset -> SageAttention2, fp16 accum).
pub fn parse_attention_impl(v: Option<&str>) -> Result<AttentionImpl> {
    match v.map(str::trim) {
        None | Some("") | Some("2") => Ok(AttentionImpl::Sage2(PvAccum::F16)),
        Some("2f32") => Ok(AttentionImpl::Sage2(PvAccum::F32)),
        Some("1") => Ok(AttentionImpl::Sage1),
        Some(o) => anyhow::bail!("QIR_SAGE={o:?}: expected 1 (SageAttention v1), 2 (SA2, fp16 PV accum) or 2f32 (SA2, fp32 PV accum)"),
    }
}

/// The attention selected by the `QIR_SAGE` environment variable, parsed once
/// per process (an unknown value is an error on every call, never a silent
/// fallback). Logged once at info level.
pub fn attention_impl() -> Result<AttentionImpl> {
    static IMPL: OnceLock<std::result::Result<AttentionImpl, String>> = OnceLock::new();
    let r = IMPL.get_or_init(|| {
        let v = std::env::var("QIR_SAGE").ok();
        let r = parse_attention_impl(v.as_deref()).map_err(|e| e.to_string());
        if let Ok(a) = &r {
            tracing::info!(attention = ?a, qir_sage = ?v, "attention implementation");
        }
        r
    });
    r.clone().map_err(anyhow::Error::msg)
}

/// Which attention operand a [`Sage2Qk`] is; fixes the per-thread scale layout
/// the kernel indexes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Role {
    /// 8 scales per 32-token warp block (`tok % 8`), padded to whole 128-token
    /// CTAs: `ceil(n/128) * 32` scales per head.
    Query,
    /// Smoothed (key mean subtracted); 4 scales per 64-token block
    /// (`(tok % 8) / 2`): `ceil(n/64) * 4` scales per head.
    Key,
}

impl Role {
    fn scales_per_head(self, n: usize) -> usize {
        match self {
            Role::Query => n.div_ceil(128) * 32,
            Role::Key => n.div_ceil(64) * 4,
        }
    }
    /// f32 scratch for the key-mean partial sums (Key only).
    fn scratch_per_head(self, n: usize) -> usize {
        match self {
            Role::Query => 0,
            Role::Key => n.div_ceil(CHUNK) * D,
        }
    }
}

/// Value object: one INT8 attention operand for the SA2 kernel. `packed` is a
/// U8 tensor: int8 `(b, h, n, 128)` HND-contiguous, then the per-thread f32
/// scales `(b, h, scales_per_head)`, then (Key) the f32 key-mean scratch. The
/// scratch lives inside the result allocation, so it outlives the async
/// kernels that use it. A distinct type from v1's `QuantizedQk` — the two
/// kernels index scales differently and must never be mixed.
pub struct Sage2Qk {
    packed: Tensor,
    b: usize,
    h: usize,
    n: usize,
    role: Role,
}

impl Sage2Qk {
    fn int8_bytes(b: usize, h: usize, n: usize) -> usize {
        b * h * n * D
    }
    fn packed_bytes(b: usize, h: usize, n: usize, role: Role) -> usize {
        Self::int8_bytes(b, h, n) + 4 * b * h * (role.scales_per_head(n) + role.scratch_per_head(n))
    }
    /// Int8 + scale bytes (scratch excluded) — for bit-exact tests.
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        let used = Self::int8_bytes(self.b, self.h, self.n)
            + 4 * self.b * self.h * self.role.scales_per_head(self.n);
        Ok(self.packed.narrow(0, 0, used)?.to_vec1::<u8>()?)
    }
}

/// Value object: V quantized to FP8 for the SA2 kernel. `packed` (U8): fp8
/// e4m3 `(b, h, 128, lpad)` (tokens transposed onto the last axis with the
/// fp8-mma 16-token permute, `lpad = ceil(n/64)*64`, padding zero), then the
/// per-channel f32 scales `(b, h, 128)`, then the f32 amax scratch. Carries the
/// [`PvAccum`] its range was chosen for — the attention takes the accumulation
/// mode FROM here, so V's scale_max and the kernel can never disagree.
pub struct Sage2V {
    packed: Tensor,
    b: usize,
    h: usize,
    n: usize,
    accum: PvAccum,
}

impl Sage2V {
    fn lpad(n: usize) -> usize {
        n.div_ceil(V_TILE) * V_TILE
    }
    fn fp8_bytes(b: usize, h: usize, n: usize) -> usize {
        b * h * D * Self::lpad(n)
    }
    fn packed_bytes(b: usize, h: usize, n: usize) -> usize {
        Self::fp8_bytes(b, h, n) + 4 * b * h * D + 4 * b * h * n.div_ceil(CHUNK) * D
    }
    /// Fp8 + scale bytes (scratch excluded) — for bit-exact tests.
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        let used = Self::fp8_bytes(self.b, self.h, self.n) + 4 * self.b * self.h * D;
        Ok(self.packed.narrow(0, 0, used)?.to_vec1::<u8>()?)
    }
}

/// Fused RoPE + (Key: smoothing) + per-thread INT8 quant of a PRE-rope bf16
/// `(B,S,H,D)` view (start_offset + strides honored).
struct RopeQuant2 {
    role: Role,
}

impl candle_core::CustomOp3 for RopeQuant2 {
    fn name(&self) -> &'static str {
        "sage2-rope-quant"
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
        candle_core::bail!("sage2-rope-quant is CUDA-only")
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
        let (b, n, h) = check_qk_view(xl, "sage2-rope-quant")?;
        let dhalf = check_rope_tables(cl, sl, n, "sage2-rope-quant")?;
        let st = xl.stride();
        let x_s = x.as_cuda_slice::<half::bf16>()?;
        let cos_s = cos.as_cuda_slice::<half::bf16>()?;
        let sin_s = sin.as_cuda_slice::<half::bf16>()?;
        let stream = dev.cuda_stream();
        let total = Sage2Qk::packed_bytes(b, h, n, self.role);
        let i8_len = Sage2Qk::int8_bytes(b, h, n);
        let scr_off = i8_len + 4 * b * h * self.role.scales_per_head(n);
        let out = unsafe { dev.alloc::<u8>(total)? };
        let rc;
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
            let part = match self.role {
                Role::Query => std::ptr::null_mut(),
                Role::Key => (op as usize + scr_off) as *mut f32,
            };
            rc = unsafe {
                sage2_rope_quant_launch(
                    xp as *const c_void,
                    cp as *const c_void,
                    sp as *const c_void,
                    op as *mut c_void,
                    scp,
                    part,
                    b as i32,
                    h as i32,
                    n as i32,
                    st[0] as u32,
                    st[1] as u32,
                    st[2] as u32,
                    dhalf as u32,
                    (self.role == Role::Key) as i32,
                    stream.cu_stream() as *mut c_void,
                )
            };
        }
        check_rc(rc, "sage2_rope_quant_launch")?;
        Ok((CudaStorage::wrap_cuda_slice(out, dev), total.into()))
    }
}

/// Per-channel FP8 quant + transpose of an f16 `(B,S,H,D)` V view.
struct QuantV2 {
    accum: PvAccum,
}

impl candle_core::CustomOp1 for QuantV2 {
    fn name(&self) -> &'static str {
        "sage2-quant-v"
    }

    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> candle_core::Result<(CpuStorage, Shape)> {
        candle_core::bail!("sage2-quant-v is CUDA-only")
    }

    fn cuda_fwd(&self, v: &CudaStorage, vl: &Layout) -> candle_core::Result<(CudaStorage, Shape)> {
        let dev = v.device().clone();
        // same D / stride-1 / 16-byte alignment contract as q/k (f16 = 2 bytes too)
        let (b, n, h) = check_qk_view(vl, "sage2-quant-v")?;
        let st = vl.stride();
        let v_s = v.as_cuda_slice::<half::f16>()?;
        let stream = dev.cuda_stream();
        let total = Sage2V::packed_bytes(b, h, n);
        let f8_len = Sage2V::fp8_bytes(b, h, n);
        let lpad = Sage2V::lpad(n);
        let out = unsafe { dev.alloc::<u8>(total)? };
        let rc;
        {
            let (vp, _a) = v_s.device_ptr(&stream);
            let (op, _b) = out.device_ptr(&stream);
            let vp = vp as usize + vl.start_offset() * std::mem::size_of::<half::f16>();
            let vsp = (op as usize + f8_len) as *mut f32;
            let part = (op as usize + f8_len + 4 * b * h * D) as *mut f32;
            rc = unsafe {
                sage2_quant_v_launch(
                    vp as *const c_void,
                    op as *mut c_void,
                    vsp,
                    part,
                    b as i32,
                    h as i32,
                    n as i32,
                    lpad as i32,
                    self.accum.scale_max(),
                    st[0] as u32,
                    st[1] as u32,
                    st[2] as u32,
                    stream.cu_stream() as *mut c_void,
                )
            };
        }
        check_rc(rc, "sage2_quant_v_launch")?;
        Ok((CudaStorage::wrap_cuda_slice(out, dev), total.into()))
    }
}

/// SA2 attention over pre-quantized operands (packed q, k, v buffers).
struct Attn2 {
    scale: f32,
    causal: bool,
    accum: PvAccum,
    b: usize,
    hq: usize,
    sq: usize,
    hk: usize,
    skv: usize,
}

impl candle_core::CustomOp3 for Attn2 {
    fn name(&self) -> &'static str {
        "sage2-attn"
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
        candle_core::bail!("sage2-attn is CUDA-only")
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
        let q_len = Sage2Qk::packed_bytes(b, hq, sq, Role::Query);
        let k_len = Sage2Qk::packed_bytes(b, hk, skv, Role::Key);
        let v_len = Sage2V::packed_bytes(b, hk, skv);
        for (l, want, what) in [(ql, q_len, "q"), (kl, k_len, "k"), (vl, v_len, "v")] {
            if l.start_offset() != 0 || l.shape().elem_count() != want {
                candle_core::bail!("sage2-attn: packed {what} size/offset mismatch vs dims");
            }
        }
        let q_s = q.as_cuda_slice::<u8>()?;
        let k_s = k.as_cuda_slice::<u8>()?;
        let v_s = v.as_cuda_slice::<u8>()?;
        let stream = dev.cuda_stream();
        let o = unsafe { dev.alloc::<half::bf16>(b * sq * hq * D)? };
        let rc;
        {
            let (qp, _a) = q_s.device_ptr(&stream);
            let (kp, _b) = k_s.device_ptr(&stream);
            let (vp, _c) = v_s.device_ptr(&stream);
            let (op, _d) = o.device_ptr(&stream);
            let qsp = (qp as usize + Sage2Qk::int8_bytes(b, hq, sq)) as *const f32;
            let ksp = (kp as usize + Sage2Qk::int8_bytes(b, hk, skv)) as *const f32;
            let vsp = (vp as usize + Sage2V::fp8_bytes(b, hk, skv)) as *const f32;
            rc = unsafe {
                sage2_attn_launch(
                    qp as *const c_void,
                    kp as *const c_void,
                    vp as *const c_void,
                    op as *mut c_void,
                    qsp,
                    ksp,
                    vsp,
                    b as i32,
                    hq as i32,
                    hk as i32,
                    sq as i32,
                    skv as i32,
                    Sage2V::lpad(skv) as i32,
                    (sq * hq * D) as u32, // O: fresh contiguous (B, sq, Hq, D)
                    (hq * D) as u32,
                    D as u32,
                    self.scale,
                    self.causal as i32,
                    (self.accum == PvAccum::F16) as i32,
                    stream.cu_stream() as *mut c_void,
                )
            };
        }
        check_rc(rc, "sage2_attn_launch")?;
        Ok((CudaStorage::wrap_cuda_slice(o, dev), (b, sq, hq, D).into()))
    }
}

/// Fused RoPE + per-thread INT8 quant (Key: smoothed with the view's own key
/// mean) of the PRE-rope bf16 view `x (B,S,H,D)`; `cos,sin (S, D/2)` are the
/// rows for exactly the view's tokens.
pub fn rope_quant(x: &Tensor, cos: &Tensor, sin: &Tensor, role: Role) -> Result<Sage2Qk> {
    let (b, n, h, _) = x.dims4()?;
    let packed = x.apply_op3(cos, sin, RopeQuant2 { role })?;
    Ok(Sage2Qk {
        packed,
        b,
        h,
        n,
        role,
    })
}

/// Per-channel FP8 quant of the f16 `(B,S,H,D)` V view for `accum`.
pub fn quant_v(v: &Tensor, accum: PvAccum) -> Result<Sage2V> {
    if v.dtype() != DType::F16 {
        anyhow::bail!("sage2::quant_v: V must be f16, got {:?}", v.dtype());
    }
    let (b, n, h, _) = v.dims4()?;
    let packed = v.apply_op1(QuantV2 { accum })?;
    Ok(Sage2V {
        packed,
        b,
        h,
        n,
        accum,
    })
}

/// SA2 attention: returns `(B, q.n, Hq, D)` bf16. The PV accumulation mode is
/// the one `v` was quantized for.
pub fn attention(q: &Sage2Qk, k: &Sage2Qk, v: &Sage2V, scale: f32, causal: bool) -> Result<Tensor> {
    if q.role != Role::Query || k.role != Role::Key {
        anyhow::bail!("sage2::attention: need (Query, Key) operands");
    }
    if q.b != k.b || k.b != v.b || k.h != v.h || k.n != v.n || !q.h.is_multiple_of(k.h) {
        anyhow::bail!(
            "sage2::attention: operand dims disagree or q heads not a multiple of kv heads (q b{} h{} n{}, k b{} h{} n{}, v b{} h{} n{})",
            q.b,
            q.h,
            q.n,
            k.b,
            k.h,
            k.n,
            v.b,
            v.h,
            v.n
        );
    }
    let op = Attn2 {
        scale,
        causal,
        accum: v.accum,
        b: q.b,
        hq: q.h,
        sq: q.n,
        hk: k.h,
        skv: k.n,
    };
    Ok(q.packed.apply_op3(&k.packed, &v.packed, op)?)
}

/// The DiT's block-causal attention with SA2. `qh,kh` PRE-rope `(B,S,H,D)`
/// bf16, `vf (B,S,H,D)` f16, `cos,sin (S, D/2)`. Text queries `[0,txt)` attend
/// causally to the text prefix; image queries attend to the whole sequence.
/// Every operand is quantized on its own S-axis narrow (zero-copy view), so
/// scales start at that call's first token and K is smoothed with that call's
/// own key mean. Returns `(B,S,H,D)` bf16.
#[allow(clippy::too_many_arguments)]
pub fn attend_block_causal(
    qh: &Tensor,
    kh: &Tensor,
    vf: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    txt_len: usize,
    scale: f32,
    accum: PvAccum,
) -> Result<Tensor> {
    let l = quant_layer(qh, kh, vf, cos, sin, txt_len, accum)?;
    let ot = attention(&l.qt, &l.kt, &l.vt, scale, true)?; // (B,txt,H,D)
    let oi = attention(&l.qi, &l.kf, &l.vf, scale, false)?; // (B,img,H,D)
    Ok(Tensor::cat(&[ot, oi], 1)?)
}

/// [`attend_block_causal`] with each operand quantized by its own per-op
/// launch ([`rope_quant`] / [`quant_v`], ten launches) — the byte-exact
/// oracle for [`quant_layer`] in `sage-test`.
#[allow(clippy::too_many_arguments)]
pub fn attend_block_causal_unfused(
    qh: &Tensor,
    kh: &Tensor,
    vf: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    txt_len: usize,
    scale: f32,
    accum: PvAccum,
) -> Result<Tensor> {
    let (_b, s, _h, _d) = qh.dims4()?;
    let img = s - txt_len;
    let (ct, st) = (cos.narrow(0, 0, txt_len)?, sin.narrow(0, 0, txt_len)?);
    let qt = rope_quant(&qh.narrow(1, 0, txt_len)?, &ct, &st, Role::Query)?;
    let kt = rope_quant(&kh.narrow(1, 0, txt_len)?, &ct, &st, Role::Key)?;
    let vt = quant_v(&vf.narrow(1, 0, txt_len)?, accum)?;
    let ot = attention(&qt, &kt, &vt, scale, true)?; // (B,txt,H,D)

    let qi = rope_quant(
        &qh.narrow(1, txt_len, img)?,
        &cos.narrow(0, txt_len, img)?,
        &sin.narrow(0, txt_len, img)?,
        Role::Query,
    )?;
    let kf = rope_quant(kh, cos, sin, Role::Key)?;
    let vfq = quant_v(vf, accum)?;
    let oi = attention(&qi, &kf, &vfq, scale, false)?; // (B,img,H,D)
    Ok(Tensor::cat(&[ot, oi], 1)?)
}

// ---------------------------------------------------------------------------
// Fused per-layer quant (three launches for all six operands)
// ---------------------------------------------------------------------------

/// Task kinds (`S2Kind` in sage2_ffi.cu).
const S2_Q: u32 = 0;
const S2_K: u32 = 1;
const S2_V: u32 = 2;
const S2_KSUM: u32 = 3;
const S2_VAMAX: u32 = 4;
const S2_MAX_TASKS: usize = 6;
/// Partial-reduction chunks per CTA (`S2_PART_PER_CTA` in sage2_ffi.cu).
const S2_PART_PER_CTA: usize = 2;

/// One quant task (mirror of `S2Task` in sage2_ffi.cu: 6 pointers, 11 u32, 1 f32).
#[repr(C)]
#[derive(Clone, Copy)]
struct S2Task {
    inp: *const c_void,
    cos: *const c_void,
    sin: *const c_void,
    partial: *mut f32,
    out: *mut c_void,
    scale: *mut f32,
    kind: u32,
    n: u32,
    nchunk: u32,
    nblk: u32,
    ncta: u32,
    nblocks: u32,
    sbz: u32,
    sseq: u32,
    sh: u32,
    sseq_cs: u32,
    lpad: u32,
    scale_max: f32,
}

impl S2Task {
    const EMPTY: S2Task = S2Task {
        inp: std::ptr::null(),
        cos: std::ptr::null(),
        sin: std::ptr::null(),
        partial: std::ptr::null_mut(),
        out: std::ptr::null_mut(),
        scale: std::ptr::null_mut(),
        kind: 0,
        n: 0,
        nchunk: 0,
        nblk: 0,
        ncta: 0,
        nblocks: 0,
        sbz: 0,
        sseq: 0,
        sh: 0,
        sseq_cs: 0,
        lpad: 0,
        scale_max: 0.0,
    };
}

/// A launch's task table (mirror of `S2Tasks`).
#[repr(C)]
struct S2Tasks {
    t: [S2Task; S2_MAX_TASKS],
    ntask: u32,
    h: u32,
    b: u32,
}

// ABI with `S2Task` / `S2Tasks` in sage2_ffi.cu (static_assert'ed there too).
const _: () = assert!(std::mem::size_of::<S2Task>() == 96);
const _: () = assert!(std::mem::size_of::<S2Tasks>() == 592);

impl S2Tasks {
    fn new(h: usize, b: usize) -> Self {
        Self {
            t: [S2Task::EMPTY; S2_MAX_TASKS],
            ntask: 0,
            h: h as u32,
            b: b as u32,
        }
    }
    fn push(&mut self, t: S2Task) {
        self.t[self.ntask as usize] = t;
        self.ntask += 1;
    }
}

/// Value object: the six quantized operands of one block-causal SA2
/// attention — the text-prefix call `(qt, kt, vt)` over `[0, txt)` and the
/// image call `(qi, kf, vf)` (image queries, full K/V). Each operand has the
/// exact payload/scale layout of the per-op [`rope_quant`] / [`quant_v`].
pub struct Sage2Layer {
    pub qt: Sage2Qk,
    pub kt: Sage2Qk,
    pub vt: Sage2V,
    pub qi: Sage2Qk,
    pub kf: Sage2Qk,
    pub vf: Sage2V,
}

fn as_cuda<'a>(st: &'a candle_core::Storage, what: &str) -> Result<&'a CudaStorage> {
    match st {
        candle_core::Storage::Cuda(c) => Ok(c),
        _ => anyhow::bail!("sage2::quant_layer: {what} must be on CUDA"),
    }
}

/// Quantize all six operands of [`attend_block_causal`] in three launches of
/// one generic kernel (V-amax partials; V quant + K-sum partials; K quant +
/// Q quant), ordered for L2 reuse. `qh,kh` PRE-rope
/// `(B,S,H,D)` bf16 views, `vf (B,S,H,D)` f16, `cos,sin (S, D/2)` bf16.
/// Byte-identical to quantizing each S-axis narrow with the per-op kernels.
#[allow(clippy::too_many_arguments)]
pub fn quant_layer(
    qh: &Tensor,
    kh: &Tensor,
    vf: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    txt_len: usize,
    accum: PvAccum,
) -> Result<Sage2Layer> {
    use candle_core::cuda_backend::cudarc::driver::CudaSlice;
    use candle_core::op::BackpropOp;
    use candle_core::Storage;

    let (b, s, h, _) = qh.dims4()?;
    if kh.dims() != qh.dims() || vf.dims() != qh.dims() {
        anyhow::bail!(
            "sage2::quant_layer: q {:?} / k {:?} / v {:?} shapes differ",
            qh.dims(),
            kh.dims(),
            vf.dims()
        );
    }
    if qh.dtype() != DType::BF16 || kh.dtype() != DType::BF16 || vf.dtype() != DType::F16 {
        anyhow::bail!("sage2::quant_layer: need q,k bf16 and v f16");
    }
    if txt_len == 0 || txt_len >= s {
        anyhow::bail!("sage2::quant_layer: need 0 < txt_len ({txt_len}) < S ({s})");
    }
    let img = s - txt_len;
    let (qs, ql) = qh.storage_and_layout();
    let (ks, kl) = kh.storage_and_layout();
    let (vs, vl) = vf.storage_and_layout();
    let (cs, cl) = cos.storage_and_layout();
    let (ss, sl) = sin.storage_and_layout();
    // The read guards stay alive until every kernel has been enqueued.
    let (qc, kc, vc, cc, sc) = (
        as_cuda(&qs, "q")?,
        as_cuda(&ks, "k")?,
        as_cuda(&vs, "v")?,
        as_cuda(&cs, "cos")?,
        as_cuda(&ss, "sin")?,
    );
    check_qk_view(ql, "sage2-quant-layer q")?;
    check_qk_view(kl, "sage2-quant-layer k")?;
    check_qk_view(vl, "sage2-quant-layer v")?;
    let dhalf = check_rope_tables(cl, sl, s, "sage2-quant-layer")?;
    let dev = qc.device().clone();
    let stream = dev.cuda_stream();

    let q_t = Sage2Qk::packed_bytes(b, h, txt_len, Role::Query);
    let q_i = Sage2Qk::packed_bytes(b, h, img, Role::Query);
    let k_t = Sage2Qk::packed_bytes(b, h, txt_len, Role::Key);
    let k_f = Sage2Qk::packed_bytes(b, h, s, Role::Key);
    let v_t = Sage2V::packed_bytes(b, h, txt_len);
    let v_f = Sage2V::packed_bytes(b, h, s);
    let alloc = |n: usize| -> Result<CudaSlice<u8>> { Ok(unsafe { dev.alloc::<u8>(n)? }) };
    let outs = [
        alloc(q_t)?,
        alloc(q_i)?,
        alloc(k_t)?,
        alloc(k_f)?,
        alloc(v_t)?,
        alloc(v_f)?,
    ];
    let rc;
    {
        let bf = std::mem::size_of::<half::bf16>();
        let (qp, _g0) = qc.as_cuda_slice::<half::bf16>()?.device_ptr(&stream);
        let (kp, _g1) = kc.as_cuda_slice::<half::bf16>()?.device_ptr(&stream);
        let (vp, _g2) = vc.as_cuda_slice::<half::f16>()?.device_ptr(&stream);
        let (cp, _g3) = cc.as_cuda_slice::<half::bf16>()?.device_ptr(&stream);
        let (sp, _g4) = sc.as_cuda_slice::<half::bf16>()?.device_ptr(&stream);
        let mut op = [0usize; 6];
        let mut _og = Vec::with_capacity(6);
        for (o, slot) in outs.iter().zip(op.iter_mut()) {
            let (p, g) = o.device_ptr(&stream);
            *slot = p as usize;
            _og.push(g);
        }
        // Element pointers of token `tok` of a (B,S,H,D) view / row `tok` of cos,sin.
        let at = |base: u64, l: &Layout, tok: usize| -> usize {
            base as usize + (l.start_offset() + tok * l.stride()[1]) * bf
        };
        let row = |base: u64, l: &Layout, tok: usize| -> usize {
            base as usize + (l.start_offset() + tok * dhalf) * bf
        };
        let (qst, kst, vst) = (ql.stride(), kl.stride(), vl.stride());
        let strides = |st: &[usize]| (st[0] as u32, st[1] as u32, st[2] as u32);
        let qk_task = |kind: u32,
                       inp: usize,
                       c: usize,
                       sn: usize,
                       out: usize,
                       n: usize,
                       st: (u32, u32, u32)| {
            let role = if kind == S2_K { Role::Key } else { Role::Query };
            let (nblk, ncta) = match role {
                Role::Query => (n.div_ceil(128) * 4, n.div_ceil(128) * 4),
                Role::Key => (n.div_ceil(64), n.div_ceil(64)),
            };
            let i8_len = Sage2Qk::int8_bytes(b, h, n);
            let scr = i8_len + 4 * b * h * role.scales_per_head(n);
            S2Task {
                inp: inp as *const c_void,
                cos: c as *const c_void,
                sin: sn as *const c_void,
                partial: if role == Role::Key {
                    (out + scr) as *mut f32
                } else {
                    std::ptr::null_mut()
                },
                out: out as *mut c_void,
                scale: (out + i8_len) as *mut f32,
                kind,
                n: n as u32,
                nchunk: n.div_ceil(CHUNK) as u32,
                nblk: nblk as u32,
                ncta: ncta as u32,
                nblocks: (ncta * h * b) as u32,
                sbz: st.0,
                sseq: st.1,
                sh: st.2,
                sseq_cs: dhalf as u32,
                lpad: 0,
                scale_max: 0.0,
            }
        };
        let v_task = |inp: usize, out: usize, n: usize, st: (u32, u32, u32)| {
            let f8 = Sage2V::fp8_bytes(b, h, n);
            let lpad = Sage2V::lpad(n);
            S2Task {
                inp: inp as *const c_void,
                partial: (out + f8 + 4 * b * h * D) as *mut f32,
                out: out as *mut c_void,
                scale: (out + f8) as *mut f32,
                kind: S2_V,
                n: n as u32,
                nchunk: n.div_ceil(CHUNK) as u32,
                ncta: (lpad / V_TILE) as u32,
                nblocks: (lpad / V_TILE * h * b) as u32,
                sbz: st.0,
                sseq: st.1,
                sh: st.2,
                lpad: lpad as u32,
                scale_max: accum.scale_max(),
                ..S2Task::EMPTY
            }
        };
        let (c0, s0) = (row(cp, cl, 0), row(sp, sl, 0));
        let (ct, stx) = (row(cp, cl, txt_len), row(sp, sl, txt_len));
        let tq = qk_task(S2_Q, at(qp, ql, 0), c0, s0, op[0], txt_len, strides(qst));
        let iq = qk_task(S2_Q, at(qp, ql, txt_len), ct, stx, op[1], img, strides(qst));
        let tk = qk_task(S2_K, at(kp, kl, 0), c0, s0, op[2], txt_len, strides(kst));
        let fk = qk_task(S2_K, at(kp, kl, 0), c0, s0, op[3], s, strides(kst));
        let tv = v_task(at(vp, vl, 0), op[4], txt_len, strides(vst));
        let fv = v_task(at(vp, vl, 0), op[5], s, strides(vst));
        // A partial pass writes into its K/V buffer's own scratch.
        let partial = |t: &S2Task, kind: u32| S2Task {
            kind,
            ncta: 0,
            nblocks: (t.nchunk as usize * h * b).div_ceil(S2_PART_PER_CTA) as u32,
            ..*t
        };
        let table = |ts: &[S2Task]| {
            let mut t = S2Tasks::new(h, b);
            for x in ts {
                t.push(*x);
            }
            t
        };
        // Launch order = L2 reuse: V was written last (to_v), then K, then Q;
        // each tensor's quant pass directly follows its partial pass, so its
        // second read hits L2 (one 4117-token K or V is 34 MB of the 72 MB L2).
        // Launch order = L2 reuse (measured best of seven orders): V was
        // written last (to_v), then K, then Q. V's amax partial reads V while
        // it is still in L2, V's quant follows it and runs alongside K's sum
        // partial, then K's quant (K just read) runs alongside Q.
        let tables = [
            table(&[partial(&fv, S2_VAMAX), partial(&tv, S2_VAMAX)]),
            table(&[fv, tv, partial(&fk, S2_KSUM), partial(&tk, S2_KSUM)]),
            table(&[fk, tk, iq, tq]),
        ];
        rc = unsafe {
            sage2_quant_layer_launch(
                tables.as_ptr(),
                tables.len() as i32,
                stream.cu_stream() as *mut c_void,
            )
        };
    }
    check_rc(rc, "sage2_quant_layer_launch")?;
    drop((qs, ks, vs, cs, ss));
    let [o_qt, o_qi, o_kt, o_kf, o_vt, o_vf] = outs;
    let wrap = |o: CudaSlice<u8>| -> Tensor {
        let n = o.len();
        Tensor::from_storage(
            Storage::Cuda(CudaStorage::wrap_cuda_slice(o, dev.clone())),
            n,
            BackpropOp::none(),
            false,
        )
    };
    let qk = |packed: Tensor, n: usize, role: Role| Sage2Qk {
        packed,
        b,
        h,
        n,
        role,
    };
    let vv = |packed: Tensor, n: usize| Sage2V {
        packed,
        b,
        h,
        n,
        accum,
    };
    Ok(Sage2Layer {
        qt: qk(wrap(o_qt), txt_len, Role::Query),
        kt: qk(wrap(o_kt), txt_len, Role::Key),
        vt: vv(wrap(o_vt), txt_len),
        qi: qk(wrap(o_qi), img, Role::Query),
        kf: qk(wrap(o_kf), s, Role::Key),
        vf: vv(wrap(o_vf), s),
    })
}

// ---------------------------------------------------------------------------
// Self-tests (sage-test)
// ---------------------------------------------------------------------------

/// f32 softmax attention on `(B,S,H,D)` operands (any dtype), returns
/// `(B,Sq,H,D)` f32. `causal`: query i sees keys j <= i (both from index 0).
fn ref_attention(q: &Tensor, k: &Tensor, v: &Tensor, scale: f64, causal: bool) -> Result<Tensor> {
    use candle_core::D as Dim;
    let t = |x: &Tensor| -> Result<Tensor> {
        Ok(x.to_dtype(DType::F32)?.transpose(1, 2)?.contiguous()?)
    };
    let (qf, kf, vf) = (t(q)?, t(k)?, t(v)?);
    let mut sc = (qf.matmul(&kf.transpose(Dim::Minus1, Dim::Minus2)?)? * scale)?;
    if causal {
        let (sq, sk) = (qf.dim(2)?, kf.dim(2)?);
        let mut m = vec![0f32; sq * sk];
        for i in 0..sq {
            for j in (i + 1)..sk {
                m[i * sk + j] = f32::NEG_INFINITY;
            }
        }
        sc = sc.broadcast_add(&Tensor::from_vec(m, (1, 1, sq, sk), q.device())?)?;
    }
    let p = candle_nn::ops::softmax_last_dim(&sc)?;
    Ok(p.matmul(&vf)?.transpose(1, 2)?.contiguous()?)
}

/// Result of [`self_test`]; every check must pass (see [`Sage2Report::ok`]).
pub struct Sage2Report {
    /// SA2 (fp16 accum) block-causal vs f32 reference, worst of B=1,2: cosine.
    pub cos_f16: f32,
    /// SA2 (fp32 accum) — same.
    pub cos_f32: f32,
    /// SageAttention v1 on the same inputs (context, not a gate).
    pub cos_v1: f32,
    /// non-finite outputs in the accuracy runs (inputs include a zeroed V
    /// channel and a zeroed K head-dim).
    pub nonfinite: usize,
    /// B=2 lanes vs B=1 runs of the same lanes (lane 1 = nonzero batch
    /// offset): differing output elements.
    pub lane_mismatches: usize,
    /// S-offset views vs fresh copies: differing quantized bytes + outputs.
    pub offset_mismatches: usize,
    /// NaN-poisoned smem partial tile (txt=21): NaN count and maxabs vs clean.
    pub poison_nan: usize,
    pub poison_maxabs: f32,
    /// run twice: differing output elements.
    pub nondeterministic: usize,
    /// fused [`quant_layer`] vs the per-op kernels: differing payload/scale
    /// bytes of the six operands + differing attention outputs (B=1, B=2,
    /// S-offset views, txt=37 and 21, fp16 and fp32 accum).
    pub fused_mismatches: usize,
}

impl Sage2Report {
    pub fn ok(&self) -> bool {
        self.cos_f16 >= 0.999
            && self.cos_f32 >= 0.999
            && self.nonfinite == 0
            && self.lane_mismatches == 0
            && self.offset_mismatches == 0
            && self.poison_nan == 0
            && self.poison_maxabs == 0.0
            && self.nondeterministic == 0
            && self.fused_mismatches == 0
    }
}

fn count_diff(a: &Tensor, b: &Tensor) -> Result<usize> {
    let a: Vec<u16> = a
        .flatten_all()?
        .to_dtype(DType::BF16)?
        .to_vec1::<half::bf16>()?
        .iter()
        .map(|x| x.to_bits())
        .collect();
    let b: Vec<u16> = b
        .flatten_all()?
        .to_dtype(DType::BF16)?
        .to_vec1::<half::bf16>()?
        .iter()
        .map(|x| x.to_bits())
        .collect();
    if a.len() != b.len() {
        anyhow::bail!("count_diff: length {} vs {}", a.len(), b.len());
    }
    Ok(a.iter().zip(&b).filter(|(x, y)| x != y).count())
}

fn byte_diff(a: &[u8], b: &[u8]) -> usize {
    a.iter().zip(b).filter(|(x, y)| x != y).count() + a.len().abs_diff(b.len())
}

/// Fused [`quant_layer`] vs the per-op [`rope_quant`] / [`quant_v`] on the
/// same views: differing bytes over the six operands + differing outputs of
/// [`attend_block_causal`] vs [`attend_block_causal_unfused`].
#[allow(clippy::too_many_arguments)]
fn fused_vs_unfused(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    txt: usize,
    scale: f32,
    accum: PvAccum,
) -> Result<usize> {
    let s = q.dim(1)?;
    let img = s - txt;
    let l = quant_layer(q, k, v, cos, sin, txt, accum)?;
    let (ct, st) = (cos.narrow(0, 0, txt)?, sin.narrow(0, 0, txt)?);
    let (ci, si) = (cos.narrow(0, txt, img)?, sin.narrow(0, txt, img)?);
    let qt = rope_quant(&q.narrow(1, 0, txt)?, &ct, &st, Role::Query)?;
    let kt = rope_quant(&k.narrow(1, 0, txt)?, &ct, &st, Role::Key)?;
    let vt = quant_v(&v.narrow(1, 0, txt)?, accum)?;
    let qi = rope_quant(&q.narrow(1, txt, img)?, &ci, &si, Role::Query)?;
    let kf = rope_quant(k, cos, sin, Role::Key)?;
    let vf = quant_v(v, accum)?;
    let mut bad = 0;
    for (a, r) in [(&l.qt, &qt), (&l.kt, &kt), (&l.qi, &qi), (&l.kf, &kf)] {
        bad += byte_diff(&a.to_bytes()?, &r.to_bytes()?);
    }
    for (a, r) in [(&l.vt, &vt), (&l.vf, &vf)] {
        bad += byte_diff(&a.to_bytes()?, &r.to_bytes()?);
    }
    let o = attend_block_causal(q, k, v, cos, sin, txt, scale, accum)?;
    let r = attend_block_causal_unfused(q, k, v, cos, sin, txt, scale, accum)?;
    Ok(bad + count_diff(&o, &r)?)
}

fn count_nonfinite(t: &Tensor) -> Result<usize> {
    let v: Vec<f32> = t.flatten_all()?.to_dtype(DType::F32)?.to_vec1()?;
    Ok(v.iter().filter(|x| !x.is_finite()).count())
}

/// SA2 self-test on the DiT's exact block-causal split with partial tiles
/// (txt=37, S=293 = 4*64+37), real RoPE tables, a strong per-channel K bias
/// (what K smoothing removes), a zeroed V channel and a zeroed K head-dim.
pub fn self_test() -> Result<Sage2Report> {
    use candle_core::Device;
    let (h, s, d, txt) = (4usize, 293usize, D, 37usize);
    let dev = Device::new_cuda(0)?;
    dev.set_seed(23)?;
    let scale = 1.0 / (d as f64).sqrt();
    let sc = scale as f32;
    // RoPE tables from random angles (|cos|,|sin| <= 1 like the model's).
    let ang = (Tensor::rand(0f32, 1f32, (s, d / 2), &dev)? * (2.0 * std::f64::consts::PI))?;
    let cos = ang.cos()?.to_dtype(DType::BF16)?;
    let sin = ang.sin()?.to_dtype(DType::BF16)?;
    let zero_at = |i: usize| -> Result<Tensor> {
        let mut m = vec![1f32; d];
        m[i] = 0.0;
        Ok(Tensor::from_vec(m, (1, 1, 1, d), &dev)?)
    };
    let k_mask = zero_at(7)?;
    let v_mask = zero_at(5)?;
    let kbias = (Tensor::randn(0f32, 1f32, (1, 1, 1, d), &dev)? * 4.0)?;
    let mk = |b: usize| -> Result<(Tensor, Tensor, Tensor)> {
        let q = Tensor::randn(0f32, 1f32, (b, s, h, d), &dev)?.to_dtype(DType::BF16)?;
        let k = Tensor::randn(0f32, 1f32, (b, s, h, d), &dev)?
            .broadcast_add(&kbias)?
            .broadcast_mul(&k_mask)?
            .to_dtype(DType::BF16)?;
        let v = Tensor::randn(0f32, 1f32, (b, s, h, d), &dev)?
            .broadcast_mul(&v_mask)?
            .to_dtype(DType::F16)?;
        Ok((q, k, v))
    };
    let reference = |q: &Tensor, k: &Tensor, v: &Tensor| -> Result<Tensor> {
        let qr = crate::rope::rope_i_bshd(q, &cos, &sin)?;
        let kr = crate::rope::rope_i_bshd(k, &cos, &sin)?;
        let ot = ref_attention(
            &qr.narrow(1, 0, txt)?,
            &kr.narrow(1, 0, txt)?,
            &v.narrow(1, 0, txt)?,
            scale,
            true,
        )?;
        let oi = ref_attention(&qr.narrow(1, txt, s - txt)?, &kr, v, scale, false)?;
        Ok(Tensor::cat(&[ot, oi], 1)?)
    };
    let cosine = |a: &Tensor, b: &Tensor| -> Result<f32> {
        crate::sage::nan_aware_compare(a, b).map(|(c, _)| c)
    };
    let v1 = |q: &Tensor, k: &Tensor, v: &Tensor| -> Result<Tensor> {
        use crate::sage::{rope_quant_bshd, sage_attention_quantized, QkRole};
        let (ct, st) = (cos.narrow(0, 0, txt)?, sin.narrow(0, 0, txt)?);
        let qt = rope_quant_bshd(&q.narrow(1, 0, txt)?, &ct, &st, QkRole::Query)?;
        let kt = rope_quant_bshd(&k.narrow(1, 0, txt)?, &ct, &st, QkRole::Key)?;
        let ot = sage_attention_quantized(&qt, &kt, &v.narrow(1, 0, txt)?, sc, true)?;
        let qi = rope_quant_bshd(
            &q.narrow(1, txt, s - txt)?,
            &cos.narrow(0, txt, s - txt)?,
            &sin.narrow(0, txt, s - txt)?,
            QkRole::Query,
        )?;
        let kf = rope_quant_bshd(k, &cos, &sin, QkRole::Key)?;
        let oi = sage_attention_quantized(&qi, &kf, v, sc, false)?;
        Ok(Tensor::cat(&[ot, oi], 1)?)
    };

    let mut rep = Sage2Report {
        cos_f16: 1.0,
        cos_f32: 1.0,
        cos_v1: 1.0,
        nonfinite: 0,
        lane_mismatches: 0,
        offset_mismatches: 0,
        poison_nan: 0,
        poison_maxabs: 0.0,
        nondeterministic: 0,
        fused_mismatches: 0,
    };

    // 1. accuracy vs f32, B=1 and B=2; 2. lanes; 5. determinism.
    let (q2, k2, v2) = mk(2)?;
    for b in [1usize, 2] {
        let (q, k, v) = if b == 2 {
            (q2.clone(), k2.clone(), v2.clone())
        } else {
            mk(1)?
        };
        let r = reference(&q, &k, &v)?;
        for accum in [PvAccum::F16, PvAccum::F32] {
            let o = attend_block_causal(&q, &k, &v, &cos, &sin, txt, sc, accum)?;
            rep.nonfinite += count_nonfinite(&o)?;
            let c = cosine(&o.to_dtype(DType::F32)?, &r)?;
            match accum {
                PvAccum::F16 => rep.cos_f16 = rep.cos_f16.min(c),
                PvAccum::F32 => rep.cos_f32 = rep.cos_f32.min(c),
            }
        }
        rep.cos_v1 = rep
            .cos_v1
            .min(cosine(&v1(&q, &k, &v)?.to_dtype(DType::F32)?, &r)?);
    }
    let o2 = attend_block_causal(&q2, &k2, &v2, &cos, &sin, txt, sc, PvAccum::F16)?;
    let o2b = attend_block_causal(&q2, &k2, &v2, &cos, &sin, txt, sc, PvAccum::F16)?;
    rep.nondeterministic += count_diff(&o2, &o2b)?;
    for lane in 0..2 {
        // lane 1's views start at a nonzero batch offset (zero-copy narrow).
        let o1 = attend_block_causal(
            &q2.narrow(0, lane, 1)?,
            &k2.narrow(0, lane, 1)?,
            &v2.narrow(0, lane, 1)?,
            &cos,
            &sin,
            txt,
            sc,
            PvAccum::F16,
        )?;
        rep.lane_mismatches += count_diff(&o1, &o2.narrow(0, lane, 1)?)?;
    }

    // 6. fused per-layer quant vs the per-op kernels (B=2 here; B=1, offset
    //    views and txt=21 below).
    for accum in [PvAccum::F16, PvAccum::F32] {
        rep.fused_mismatches += fused_vs_unfused(&q2, &k2, &v2, &cos, &sin, txt, sc, accum)?;
    }
    for lane in 0..2 {
        rep.fused_mismatches += fused_vs_unfused(
            &q2.narrow(0, lane, 1)?,
            &k2.narrow(0, lane, 1)?,
            &v2.narrow(0, lane, 1)?,
            &cos,
            &sin,
            txt,
            sc,
            PvAccum::F16,
        )?;
    }
    rep.fused_mismatches += fused_vs_unfused(&q2, &k2, &v2, &cos, &sin, 21, sc, PvAccum::F16)?;

    // 3. S-axis offset views vs fresh copies, every op.
    {
        let pad = 5usize;
        let big = |dt: DType| -> Result<Tensor> {
            Ok(Tensor::randn(0f32, 1f32, (1, s + pad, h, d), &dev)?.to_dtype(dt)?)
        };
        let (qb, kb, vb) = (big(DType::BF16)?, big(DType::BF16)?, big(DType::F16)?);
        let (qv, kv, vv) = (
            qb.narrow(1, pad, s)?,
            kb.narrow(1, pad, s)?,
            vb.narrow(1, pad, s)?,
        );
        let (qc, kc, vc) = (qv.copy()?, kv.copy()?, vv.copy()?);
        let bytes = |a: Vec<u8>, b: Vec<u8>| {
            a.iter().zip(&b).filter(|(x, y)| x != y).count() + a.len().abs_diff(b.len())
        };
        let qa = rope_quant(&qv, &cos, &sin, Role::Query)?;
        let qr = rope_quant(&qc, &cos, &sin, Role::Query)?;
        let ka = rope_quant(&kv, &cos, &sin, Role::Key)?;
        let kr = rope_quant(&kc, &cos, &sin, Role::Key)?;
        let va = quant_v(&vv, PvAccum::F16)?;
        let vr = quant_v(&vc, PvAccum::F16)?;
        rep.offset_mismatches += bytes(qa.to_bytes()?, qr.to_bytes()?);
        rep.offset_mismatches += bytes(ka.to_bytes()?, kr.to_bytes()?);
        rep.offset_mismatches += bytes(va.to_bytes()?, vr.to_bytes()?);
        rep.fused_mismatches += fused_vs_unfused(&qv, &kv, &vv, &cos, &sin, txt, sc, PvAccum::F16)?;
        let oa = attend_block_causal(&qv, &kv, &vv, &cos, &sin, txt, sc, PvAccum::F16)?;
        let or = attend_block_causal(&qc, &kc, &vc, &cos, &sin, txt, sc, PvAccum::F16)?;
        rep.offset_mismatches += count_diff(&oa, &or)?;
    }

    // 4. NaN-poisoned shared memory, partial tiles (txt=21 < 64, S=293).
    {
        let txt = 21usize;
        let (q, k, v) = mk(1)?;
        let (ct, st) = (cos.narrow(0, 0, txt)?, sin.narrow(0, 0, txt)?);
        let qt = rope_quant(&q.narrow(1, 0, txt)?, &ct, &st, Role::Query)?;
        let kt = rope_quant(&k.narrow(1, 0, txt)?, &ct, &st, Role::Key)?;
        let vt = quant_v(&v.narrow(1, 0, txt)?, PvAccum::F16)?;
        let qi = rope_quant(
            &q.narrow(1, txt, s - txt)?,
            &cos.narrow(0, txt, s - txt)?,
            &sin.narrow(0, txt, s - txt)?,
            Role::Query,
        )?;
        let kf = rope_quant(&k, &cos, &sin, Role::Key)?;
        let vq = quant_v(&v, PvAccum::F16)?;
        let ct_ = attention(&qt, &kt, &vt, sc, true)?;
        let ci = attention(&qi, &kf, &vq, sc, false)?;
        crate::sage::poison_smem(&dev)?;
        let pt = attention(&qt, &kt, &vt, sc, true)?;
        crate::sage::poison_smem(&dev)?;
        let pi = attention(&qi, &kf, &vq, sc, false)?;
        for (c, p) in [(&ct_, &pt), (&ci, &pi)] {
            let cv: Vec<f32> = c.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
            let pv: Vec<f32> = p.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
            for (x, y) in cv.iter().zip(&pv) {
                if x.is_nan() || y.is_nan() {
                    rep.poison_nan += 1;
                } else {
                    rep.poison_maxabs = rep.poison_maxabs.max((x - y).abs());
                }
            }
        }
    }
    Ok(rep)
}
