//! Tile-config table of the ConvRot INT8 CUTLASS EVT GEMM
//! (`qwen-image-rs-gemm-merge-tune`). CPU-compiled so the selection is
//! unit-tested on every build; the kernels live in
//! `kernels/convrot/int8_gemm.cu` (`Cfg<i>`, same order).
//!
//! INT8 accumulation is exact and the dequant epilogue is per element, so
//! every config computes bit-identical outputs: the choice is speed only,
//! and it is a fixed function of the shape (deterministic, no split-K).

/// One CUTLASS tile config: threadblock and warp tile, pipeline stages and
/// threadblock swizzle (CTAs rastered in groups of `swizzle` N-tiles).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TileConfig {
    pub tb: (u16, u16, u16),
    pub warp: (u16, u16, u16),
    pub stages: u8,
    pub swizzle: u8,
}

/// The compiled configs, by index (= the `cfg` passed to the launcher).
pub const GEMM_CONFIGS: [TileConfig; 4] = [
    // 0: the original config; best (within noise) for B=1 DiT shapes.
    TileConfig {
        tb: (128, 128, 64),
        warp: (64, 64, 64),
        stages: 3,
        swizzle: 1,
    },
    // 1: swizzled 128x128 for M >= 8192 (B >= 2): L2 reuse of A.
    TileConfig {
        tb: (128, 128, 64),
        warp: (64, 64, 64),
        stages: 3,
        swizzle: 4,
    },
    // 2: small tiles for the tail linears (M = 2 / txt, N = 64).
    TileConfig {
        tb: (64, 128, 64),
        warp: (32, 64, 64),
        stages: 4,
        swizzle: 1,
    },
    // 3: 128x256 swizzled for M >= 8192 with K >= 8192 (MLP out at B >= 2).
    TileConfig {
        tb: (128, 256, 64),
        warp: (64, 64, 64),
        stages: 3,
        swizzle: 4,
    },
];

/// Below this M a GEMM is "small" (tail linears at M = 2 or the text length).
const SMALL_M: usize = 64;
/// From this M on (B >= 2 at 1024², M = B * 4117) the swizzled configs win.
const LARGE_M: usize = 8192;

/// The tile config index for an `(M, N, K)` INT8 GEMM, from `gemm-bench`
/// (RTX 4090, min of 3 interleaved rounds; HANDOVER "GEMM merge + tile
/// tuning"): small M or N -> 64x128 (~40% faster than 128x128 there);
/// M >= 8192 -> swizzle 4, with 128x256 when K >= 8192; else 128x128.
pub fn gemm_config(m: usize, n: usize, k: usize) -> u8 {
    if m <= SMALL_M || n <= 64 {
        2
    } else if m >= LARGE_M && k >= 8192 {
        3
    } else if m >= LARGE_M {
        1
    } else {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn b1_dit_shapes_keep_the_original_config() {
        for (n, k) in [(4096, 4096), (8192, 4096), (24576, 4096), (4096, 12288)] {
            assert_eq!(gemm_config(4117, n, k), 0, "N={n} K={k}");
        }
    }

    #[test]
    fn tails_take_the_small_tile() {
        assert_eq!(gemm_config(2, 16384, 4096), 2); // modulation
        assert_eq!(gemm_config(2, 4096, 256), 2); // time_embed.linear_1
        assert_eq!(gemm_config(37, 4096, 4096), 2); // txt_in at a short prompt
        assert_eq!(gemm_config(4117, 64, 4096), 2); // proj_out
    }

    #[test]
    fn batched_shapes_swizzle() {
        assert_eq!(gemm_config(8234, 4096, 4096), 1);
        assert_eq!(gemm_config(8234, 24576, 4096), 1);
        assert_eq!(gemm_config(8234, 4096, 12288), 3);
    }

    #[test]
    fn every_selected_index_is_compiled() {
        for m in [1usize, 2, 64, 65, 4117, 8192, 16468] {
            for n in [64usize, 4096, 24576] {
                for k in [256usize, 4096, 12288] {
                    assert!((gemm_config(m, n, k) as usize) < GEMM_CONFIGS.len());
                }
            }
        }
    }
}
