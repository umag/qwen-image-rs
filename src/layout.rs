//! Layout guard for the raw-pointer CUDA bridges (fusednorm / convrot / sage).
//!
//! A candle `CustomOp` receives the storage plus a [`Layout`]; the storage's
//! device pointer is the START OF THE ALLOCATION, not of the view. A dense view
//! can still begin at a nonzero `start_offset`: candle's `contiguous()` is a
//! no-op for any `is_contiguous` view, and size-1 dims are skipped in that
//! check, so e.g. a B=1 S-axis narrow `(1, S, D).narrow(1, drop, n)` stays a
//! zero-copy view at offset `drop * D`. A kernel handed the base pointer then
//! reads the wrong rows — silently. Every bridge must add the offset, which is
//! what [`dense_offset`] returns (and it rejects non-dense views outright).

use candle_core::Layout;

/// Element offset of a dense (row-major contiguous) view into its storage.
/// Bridges add `dense_offset(..)? * size_of::<T>()` to the storage pointer.
/// Errors if the view is not dense: the kernels index a flat `m * n` slab.
pub fn dense_offset(l: &Layout, what: &str) -> candle_core::Result<usize> {
    if !l.is_contiguous() {
        candle_core::bail!(
            "{what}: kernel needs a dense view, got shape {:?} strides {:?}",
            l.shape(),
            l.stride()
        );
    }
    Ok(l.start_offset())
}

#[cfg(test)]
mod tests {
    use super::dense_offset;
    use candle_core::{DType, Device, Tensor};

    #[test]
    fn fresh_tensor_has_zero_offset() {
        let t = Tensor::zeros((1, 21, 64), DType::F32, &Device::Cpu).unwrap();
        assert_eq!(dense_offset(t.layout(), "t").unwrap(), 0);
    }

    /// The generate B=1 embeds case: the S-axis narrow past the system prefix
    /// is dense (so `contiguous()` keeps it as a view) but starts at drop*D.
    #[test]
    fn b1_seq_narrow_is_dense_with_offset() {
        let (drop, n, d) = (14usize, 21usize, 64usize);
        let t = Tensor::zeros((1, drop + n, d), DType::F32, &Device::Cpu).unwrap();
        let v = t.narrow(1, drop, n).unwrap().contiguous().unwrap();
        assert_eq!(dense_offset(v.layout(), "v").unwrap(), drop * d);
    }

    #[test]
    fn b2_seq_narrow_is_rejected() {
        let t = Tensor::zeros((2, 35, 64), DType::F32, &Device::Cpu).unwrap();
        let v = t.narrow(1, 14, 21).unwrap();
        assert!(dense_offset(v.layout(), "v").is_err());
    }

    #[test]
    fn transpose_is_rejected() {
        let t = Tensor::zeros((8, 16), DType::F32, &Device::Cpu).unwrap();
        let v = t.t().unwrap();
        assert!(dense_offset(v.layout(), "v").is_err());
    }
}
