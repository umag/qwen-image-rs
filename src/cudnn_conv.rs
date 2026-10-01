//! cuDNN conv2d for the VAE decoder (feature `cudnn`).
//!
//! candle 0.11's own cuDNN conv path leaves the convolution math type at
//! `CUDNN_DEFAULT_MATH`, so for bf16 the heuristics pick FP32-SIMT kernels
//! (`implicit_convolve_sgemm`, `ampere_sgemm_*`) — slower than candle's
//! im2col + tensor-core GEMM. This bridge sets `CUDNN_TENSOR_OP_MATH`, caches
//! the chosen algorithm per shape, and allocates the workspace uninitialized.
//!
//! bf16 only, NCHW dense input, NCHW (OIHW) dense filter, stride 1, dilation 1,
//! symmetric padding. Accumulation is f32 (cuDNN's PSEUDO_BFLOAT16 config).

use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;

use candle_core::backend::BackendStorage;
use candle_core::cuda_backend::cudarc::cudnn::safe::{ConvForward, Cudnn};
use candle_core::cuda_backend::cudarc::cudnn::sys;
use candle_core::{CpuStorage, CudaStorage, DType, Layout, Shape, Tensor};

use crate::layout::dense_offset;

type Algo = sys::cudnnConvolutionFwdAlgo_t;

thread_local! {
    static HANDLE: RefCell<Option<Arc<Cudnn>>> = const { RefCell::new(None) };
    static ALGOS: RefCell<HashMap<[usize; 6], Algo>> = RefCell::new(HashMap::new());
}

fn handle(dev: &candle_core::CudaDevice) -> candle_core::Result<Arc<Cudnn>> {
    HANDLE.with(|h| {
        if let Some(c) = h.borrow().as_ref() {
            return Ok(c.clone());
        }
        let c = Cudnn::new(dev.cuda_stream()).map_err(candle_core::Error::wrap)?;
        *h.borrow_mut() = Some(c.clone());
        Ok(c)
    })
}

/// Optional override for experiments: `QIR_CUDNN_ALGO=<0..7>` (cudnn enum).
fn algo_override() -> Option<Algo> {
    let v: u32 = std::env::var("QIR_CUDNN_ALGO").ok()?.parse().ok()?;
    use sys::cudnnConvolutionFwdAlgo_t as A;
    Some(match v {
        0 => A::CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
        1 => A::CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM,
        2 => A::CUDNN_CONVOLUTION_FWD_ALGO_GEMM,
        3 => A::CUDNN_CONVOLUTION_FWD_ALGO_DIRECT,
        4 => A::CUDNN_CONVOLUTION_FWD_ALGO_FFT,
        5 => A::CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING,
        6 => A::CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD,
        _ => A::CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED,
    })
}

struct Conv2dCudnn {
    padding: usize,
}

impl candle_core::CustomOp2 for Conv2dCudnn {
    fn name(&self) -> &'static str {
        "conv2d-cudnn"
    }

    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> candle_core::Result<(CpuStorage, Shape)> {
        candle_core::bail!("conv2d-cudnn is CUDA-only")
    }

    fn cuda_fwd(
        &self,
        x: &CudaStorage,
        x_l: &Layout,
        w: &CudaStorage,
        w_l: &Layout,
    ) -> candle_core::Result<(CudaStorage, Shape)> {
        let dev = x.device().clone();
        let (b, c, h, wd) = x_l.shape().dims4()?;
        let (o, ci, kh, kw) = w_l.shape().dims4()?;
        if ci != c {
            candle_core::bail!("conv2d-cudnn: input C {c} != filter C {ci}");
        }
        let p = self.padding;
        if h + 2 * p < kh || wd + 2 * p < kw {
            candle_core::bail!("conv2d-cudnn: kernel larger than padded input");
        }
        let (oh, ow) = (h + 2 * p - kh + 1, wd + 2 * p - kw + 1);
        let xo = dense_offset(x_l, "conv2d-cudnn x")?;
        let wo = dense_offset(w_l, "conv2d-cudnn w")?;
        let xs = x.as_cuda_slice::<half::bf16>()?.slice(xo..);
        let ws = w.as_cuda_slice::<half::bf16>()?.slice(wo..);
        let cudnn = handle(&dev)?;
        let e = candle_core::Error::wrap;
        let i = |v: usize| v as i32;
        let mut conv = cudnn
            .create_conv2d::<f32>(
                [i(p), i(p)],
                [1, 1],
                [1, 1],
                sys::cudnnConvolutionMode_t::CUDNN_CROSS_CORRELATION,
            )
            .map_err(e)?;
        conv.set_math_type(sys::cudnnMathType_t::CUDNN_TENSOR_OP_MATH)
            .map_err(e)?;
        let nchw = sys::cudnnTensorFormat_t::CUDNN_TENSOR_NCHW;
        let xd = cudnn
            .create_4d_tensor::<half::bf16>(nchw, [i(b), i(c), i(h), i(wd)])
            .map_err(e)?;
        let wdsc = cudnn
            .create_4d_filter::<half::bf16>(nchw, [i(o), i(c), i(kh), i(kw)])
            .map_err(e)?;
        let yd = cudnn
            .create_4d_tensor::<half::bf16>(nchw, [i(b), i(o), i(oh), i(ow)])
            .map_err(e)?;
        let fwd = ConvForward {
            conv: &conv,
            x: &xd,
            w: &wdsc,
            y: &yd,
        };
        let key = [b, c, h, wd, o, kh * 16 + p];
        let algo = match algo_override() {
            Some(a) => a,
            None => match ALGOS.with(|m| m.borrow().get(&key).copied()) {
                Some(a) => a,
                None => {
                    let a = fwd.pick_algorithm().map_err(e)?;
                    if std::env::var_os("QIR_CUDNN_DEBUG").is_some() {
                        eprintln!("conv2d-cudnn {key:?} -> {a:?}");
                    }
                    ALGOS.with(|m| m.borrow_mut().insert(key, a));
                    a
                }
            },
        };
        let ws_bytes = fwd.get_workspace_size(algo).map_err(e)?;
        let stream = dev.cuda_stream();
        let mut workspace =
            unsafe { stream.alloc::<u8>(ws_bytes.max(1)) }.map_err(candle_core::Error::wrap)?;
        let mut out = unsafe { dev.alloc::<half::bf16>(b * o * oh * ow)? };
        unsafe {
            fwd.launch(
                algo,
                Some(&mut workspace),
                (half::bf16::ONE, half::bf16::ZERO),
                &xs,
                &ws,
                &mut out,
            )
        }
        .map_err(e)?;
        Ok((
            CudaStorage::wrap_cuda_slice(out, dev),
            Shape::from((b, o, oh, ow)),
        ))
    }
}

/// `conv2d(x, w)` (stride 1, dilation 1, `padding` on both sides) through
/// cuDNN with tensor-core math. `x` (B,C,H,W) and `w` (O,C,kh,kw), both bf16.
pub fn conv2d_bf16(x: &Tensor, w: &Tensor, padding: usize) -> candle_core::Result<Tensor> {
    if x.dtype() != DType::BF16 || w.dtype() != DType::BF16 {
        candle_core::bail!("conv2d-cudnn: bf16 only");
    }
    x.contiguous()?
        .apply_op2_no_bwd(&w.contiguous()?, &Conv2dCudnn { padding })
}

/// `cudnn-test`: the bridge vs candle's im2col conv on VAE-like shapes, plus
/// an offset-view check. Returns the number of failures (the verb exits
/// nonzero on any).
pub fn self_test(dev: &candle_core::Device) -> candle_core::Result<usize> {
    let mut fails = 0;
    // (B, C, H, W, O, k, padding)
    let cases = [
        (1, 64, 16, 16, 64, 3, 1),
        (1, 96, 8, 12, 32, 1, 0),
        (1, 144, 8, 8, 4, 3, 1),
        (2, 32, 5, 7, 48, 3, 1),
        (1, 288, 33, 31, 288, 3, 1),
    ];
    for &(b, c, h, w, o, k, p) in &cases {
        let x = Tensor::randn(0f32, 1f32, (b, c, h, w), dev)?.to_dtype(DType::BF16)?;
        let wt = (Tensor::randn(0f32, 1f32, (o, c, k, k), dev)? / ((c * k * k) as f64).sqrt())?
            .to_dtype(DType::BF16)?;
        let ours = conv2d_bf16(&x, &wt, p)?.to_dtype(DType::F32)?;
        let refr = x.conv2d(&wt, p, 1, 1, 1)?.to_dtype(DType::F32)?;
        let dot = (&ours * &refr)?.sum_all()?.to_scalar::<f32>()?;
        let na = ours.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt();
        let nb = refr.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt();
        let cos = dot / (na * nb).max(1e-30);
        let maxabs = (&ours - &refr)?.abs()?.max_all()?.to_scalar::<f32>()?;
        let ok = ours.dims() == refr.dims() && cos > 0.99999 && maxabs < 0.05;
        if !ok {
            fails += 1;
        }
        println!(
            "conv ({b},{c},{h},{w})->{o} k{k} p{p}: cos {cos:.7} maxabs {maxabs:.4} {}",
            if ok { "OK" } else { "FAIL" }
        );
    }
    // Offset view: a B=1 narrow of a (2,C,H,W) batch is a dense view at
    // start_offset C*H*W (the size-1 batch dim is skipped by is_contiguous).
    let big = Tensor::randn(0f32, 1f32, (2, 64, 12, 12), dev)?.to_dtype(DType::BF16)?;
    let view = big.narrow(0, 1, 1)?;
    let wt = (Tensor::randn(0f32, 1f32, (32, 64, 3, 3), dev)? / 24.0)?.to_dtype(DType::BF16)?;
    let offset = view.layout().start_offset();
    let a = conv2d_bf16(&view, &wt, 1)?;
    let fresh = view.to_dtype(DType::F32)?.to_dtype(DType::BF16)?; // new storage, offset 0
    let bb = conv2d_bf16(&fresh, &wt, 1)?;
    let diff = (a.to_dtype(DType::F32)? - bb.to_dtype(DType::F32)?)?
        .abs()?
        .max_all()?
        .to_scalar::<f32>()?;
    let ok = offset > 0 && diff == 0.0;
    if !ok {
        fails += 1;
    }
    println!(
        "offset view (start_offset {offset}) vs fresh copy: {}",
        if ok { "BIT-IDENTICAL" } else { "MISMATCH" }
    );
    Ok(fails)
}
