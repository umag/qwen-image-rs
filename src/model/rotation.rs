//! ConvRot rotation foundation (pytorch/ao#4695). **Phase 5 / ConvRot.**
//!
//! A group-wise **Regular** Hadamard rotation used to suppress activation
//! outliers before INT8 quantization. Regular (not Sylvester) Hadamard achieves
//! the optimal √n column discrepancy, so it *spreads* outliers instead of
//! amplifying them (Sylvester's all-ones column blows up DiT `proj_out`).
//!
//! The group is 256. For a linear `y = x Wᵀ`, insert `Rᵀ R = I` in the input
//! space: rotate both the activation and the weight by `Rᵀ` (group-wise) and
//! the output is unchanged — `(x Rᵀ)(W Rᵀ)ᵀ = x Rᵀ R Wᵀ = x Wᵀ` — but both
//! rotated tensors quantize to INT8 with far smaller error.

use candle_core::{Device, Tensor};

use crate::Result;

/// ConvRot group size.
pub const GROUP: usize = 256;

/// Orthonormal Regular Hadamard matrix of order 256, as `(256, 256)` f32.
/// Built as the 4-fold Kronecker of the order-4 regular Hadamard `H4/2`
/// (`H4` has -1 on the diagonal, +1 elsewhere: orthogonal with constant row
/// sum 2). Kronecker of orthonormal matrices is orthonormal.
pub fn regular_hadamard_256(dev: &Device) -> Result<Tensor> {
    // R4 = H4 / 2 (orthonormal, regular).
    let r4 = |i: usize, j: usize| -> f32 {
        if i == j {
            -0.5
        } else {
            0.5
        }
    };
    let mut data = vec![0f32; GROUP * GROUP];
    for i in 0..GROUP {
        for j in 0..GROUP {
            // R256[i][j] = product of R4 over the 4 base-4 digits of i, j.
            let (mut ii, mut jj) = (i, j);
            let mut v = 1.0f32;
            for _ in 0..4 {
                v *= r4(ii % 4, jj % 4);
                ii /= 4;
                jj /= 4;
            }
            data[i * GROUP + j] = v;
        }
    }
    Ok(Tensor::from_vec(data, (GROUP, GROUP), dev)?)
}

/// Group-wise rotate the last dim of `x` (…, D) by `Rᵀ`, D divisible by 256.
/// Returns `x @ Rᵀ` applied within each 256-group.
pub fn rotate(x: &Tensor, r: &Tensor) -> Result<Tensor> {
    let dims = x.dims().to_vec();
    let d = *dims.last().unwrap();
    debug_assert_eq!(d % GROUP, 0, "dim {d} not a multiple of {GROUP}");
    let n: usize = dims[..dims.len() - 1].iter().product();
    let groups = d / GROUP;
    // (N, groups, 256) -> flatten to (N*groups, 256), matmul Rᵀ, restore.
    let flat = x.reshape((n * groups, GROUP))?;
    let rot = flat.matmul(&r.t()?.to_dtype(flat.dtype())?)?;
    Ok(rot.reshape(dims)?)
}

/// Fold `Rᵀ` into a linear's weight `W (out, in)` (rotate input groups):
/// `W' = W Rᵀ`, matching `rotate` on the activation so the pair cancels.
pub fn fold_weight(w: &Tensor, r: &Tensor) -> Result<Tensor> {
    rotate(w, r)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::DType;

    #[test]
    fn hadamard_is_orthonormal() -> Result<()> {
        let dev = Device::Cpu;
        let r = regular_hadamard_256(&dev)?;
        let prod = r.matmul(&r.t()?)?; // R Rᵀ should be I
        let eye = Tensor::eye(GROUP, DType::F32, &dev)?;
        let err = (prod - eye)?.abs()?.max_all()?.to_scalar::<f32>()?;
        assert!(err < 1e-4, "R Rᵀ != I, max err {err}");
        Ok(())
    }

    #[test]
    fn regular_constant_row_sum() -> Result<()> {
        // Regular Hadamard: constant row sum. For R256 = (H4/2)^⊗4 the row sum
        // is (2/2)^4 = 1 (H4 row sum 2, /2 -> 1, product over 4 -> 1).
        let dev = Device::Cpu;
        let r = regular_hadamard_256(&dev)?;
        let sums = r.sum(1)?.to_vec1::<f32>()?;
        for s in sums {
            assert!((s - 1.0).abs() < 1e-4, "row sum {s} != 1");
        }
        Ok(())
    }

    #[test]
    fn rotation_preserves_linear_output() -> Result<()> {
        // y = x Wᵀ must equal rotate(x) @ rotate(W)ᵀ.
        let dev = Device::Cpu;
        let r = regular_hadamard_256(&dev)?;
        let x = Tensor::randn(0f32, 1f32, (3, 512), &dev)?; // 512 = 2 groups
        let w = Tensor::randn(0f32, 1f32, (128, 512), &dev)?;
        let y1 = x.matmul(&w.t()?)?;
        let xr = rotate(&x, &r)?;
        let wr = fold_weight(&w, &r)?;
        let y2 = xr.matmul(&wr.t()?)?;
        let err = (y1 - y2)?.abs()?.max_all()?.to_scalar::<f32>()?;
        assert!(err < 1e-3, "fold changed output, max err {err}");
        Ok(())
    }
}
