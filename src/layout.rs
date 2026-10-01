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

/// [`dense_offset`] in bytes for element type `T`, to add to a device pointer.
pub fn dense_byte_offset<T>(l: &Layout, what: &str) -> candle_core::Result<u64> {
    Ok((dense_offset(l, what)? * std::mem::size_of::<T>()) as u64)
}

/// A 2-D `(M, K)` view whose rows are each dense (unit inner stride) but may
/// sit `ld >= K` elements apart — e.g. a column narrow of a wider matrix, such
/// as one half of a merged `(M, 2K)` GEMM output. Returns `(start_offset, ld)`
/// in elements. For `M <= 1` the row stride is irrelevant and `ld = K`.
/// Errors on a non-unit inner stride or overlapping rows (`ld < K`).
pub fn row_strided_2d(l: &Layout, what: &str) -> candle_core::Result<(usize, usize)> {
    let (m, k) = l.shape().dims2()?;
    let st = l.stride();
    if k > 1 && st[1] != 1 {
        candle_core::bail!(
            "{what}: kernel needs unit-stride rows, got shape {:?} strides {:?}",
            l.shape(),
            st
        );
    }
    let ld = if m <= 1 { k } else { st[0] };
    if ld < k {
        candle_core::bail!("{what}: overlapping rows (row stride {ld} < K = {k})");
    }
    Ok((l.start_offset(), ld))
}

#[cfg(test)]
mod tests {
    use super::{dense_offset, row_strided_2d};
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
        assert_eq!(
            super::dense_byte_offset::<f32>(v.layout(), "v").unwrap(),
            (drop * d * 4) as u64
        );
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

    #[test]
    fn dense_matrix_is_row_strided_with_ld_k() {
        let t = Tensor::zeros((5, 32), DType::F32, &Device::Cpu).unwrap();
        assert_eq!(row_strided_2d(t.layout(), "t").unwrap(), (0, 32));
    }

    /// The merged gate|proj case: the two column halves of an (M, 2K) matrix.
    #[test]
    fn column_halves_have_ld_2k_and_offsets() {
        let (m, k) = (7usize, 16usize);
        let t = Tensor::zeros((m, 2 * k), DType::F32, &Device::Cpu).unwrap();
        let g = t.narrow(1, 0, k).unwrap();
        let p = t.narrow(1, k, k).unwrap();
        assert_eq!(row_strided_2d(g.layout(), "g").unwrap(), (0, 2 * k));
        assert_eq!(row_strided_2d(p.layout(), "p").unwrap(), (k, 2 * k));
        // A row narrow of a column half adds drop * ld.
        let pr = p.narrow(0, 3, 4).unwrap();
        assert_eq!(
            row_strided_2d(pr.layout(), "pr").unwrap(),
            (3 * 2 * k + k, 2 * k)
        );
    }

    #[test]
    fn single_row_ignores_row_stride() {
        let t = Tensor::zeros((4, 64), DType::F32, &Device::Cpu).unwrap();
        let v = t.narrow(0, 2, 1).unwrap().narrow(1, 8, 32).unwrap();
        assert_eq!(row_strided_2d(v.layout(), "v").unwrap(), (2 * 64 + 8, 32));
    }

    #[test]
    fn row_strided_rejects_transpose_and_non_2d() {
        let t = Tensor::zeros((8, 16), DType::F32, &Device::Cpu).unwrap();
        assert!(row_strided_2d(t.t().unwrap().layout(), "v").is_err());
        let t3 = Tensor::zeros((2, 8, 16), DType::F32, &Device::Cpu).unwrap();
        assert!(row_strided_2d(t3.layout(), "v3").is_err());
    }
}
