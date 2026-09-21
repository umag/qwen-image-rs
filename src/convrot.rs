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

/// End-to-end bridge self-test: candle U8(int8) tensors -> kernel -> candle I32,
/// compared to an f32 reference. Returns the max abs difference (should be 0).
pub fn self_test() -> Result<f32> {
    use candle_core::{DType, Device};
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
    let c = int8_gemm(&a_u8, &b_u8)?.to_dtype(DType::F32)?;

    let af = Tensor::from_vec(
        ai.iter().map(|&v| v as f32).collect::<Vec<f32>>(),
        (m, k),
        &dev,
    )?;
    let bf = Tensor::from_vec(
        bi.iter().map(|&v| v as f32).collect::<Vec<f32>>(),
        (n, k),
        &dev,
    )?;
    let cref = af.matmul(&bf.t()?)?;
    Ok((c - cref)?.abs()?.max_all()?.to_scalar::<f32>()?)
}
