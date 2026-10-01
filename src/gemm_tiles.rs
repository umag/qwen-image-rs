//! Tile-config table of the ConvRot INT8 CUTLASS EVT GEMM
//! (`qwen-image-rs-gemm-merge-tune`, `qwen-image-rs-gemm-tiling-push`).
//! CPU-compiled so the selection is unit-tested on every build; the kernels
//! live in `kernels/convrot/int8_gemm.cu` (`Cfg<i>`, same order).
//!
//! INT8 accumulation is exact and the dequant epilogue is per element, so
//! every config computes bit-identical outputs: the choice is speed only,
//! and it is a fixed function of the shape (deterministic). Stream-K configs
//! sum int32 partials (exact, associative) in a fixed turnstile order.

/// How a config's CTAs walk the output tiles.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Raster {
    /// `GemmIdentityThreadblockSwizzle<n>`: data-parallel, one CTA per tile,
    /// rastered in groups of `n` N-tiles.
    Tiles(u8),
    /// `ThreadblockSwizzleStreamK`: tiles' K-iterations spread evenly over
    /// the SMs, partial int32 tiles fixed up in a deterministic order.
    StreamK,
}

/// One CUTLASS tile config: threadblock and warp tile, pipeline stages and
/// CTA raster.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TileConfig {
    pub tb: (u16, u16, u16),
    pub warp: (u16, u16, u16),
    pub stages: u8,
    pub raster: Raster,
}

const fn cfg(tb: (u16, u16, u16), warp: (u16, u16, u16), stages: u8, raster: Raster) -> TileConfig {
    TileConfig {
        tb,
        warp,
        stages,
        raster,
    }
}

/// The compiled configs, by index (= the `cfg` passed to the launcher).
pub const GEMM_CONFIGS: [TileConfig; 8] = [
    // 0: the original config; best (within noise) for B=1 DiT shapes.
    cfg((128, 128, 64), (64, 64, 64), 3, Raster::Tiles(1)),
    // 1: swizzled 128x128 for M >= 8192 (B >= 2): L2 reuse of A.
    cfg((128, 128, 64), (64, 64, 64), 3, Raster::Tiles(4)),
    // 2: small tiles for the tail linears (M = 2 / txt, N = 64).
    cfg((64, 128, 64), (32, 64, 64), 4, Raster::Tiles(1)),
    // 3: 128x256 swizzled for M >= 8192 with K >= 8192 (MLP out at B >= 2).
    cfg((128, 256, 64), (64, 64, 64), 3, Raster::Tiles(4)),
    // SPIKE
    cfg((128, 128, 64), (64, 64, 64), 3, Raster::StreamK),
    cfg((128, 256, 64), (64, 64, 64), 3, Raster::StreamK),
    cfg((256, 128, 64), (64, 64, 64), 3, Raster::Tiles(1)),
    cfg((64, 64, 64), (32, 32, 64), 6, Raster::Tiles(1)),
];

/// How one GEMM is launched: `main` covers the rows up to the last full
/// `main` M-tile; with `tail = Some(t)` the remaining `M mod TBM` rows (the
/// text prefix past 4096 image tokens) run as a second launch with config
/// `t`, so the main launch has no near-empty last M-tile row of CTAs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GemmPlan {
    pub main: u8,
    pub tail: Option<u8>,
}

impl GemmPlan {
    pub const fn single(main: u8) -> Self {
        Self { main, tail: None }
    }

    /// The launches as `(first_row, rows, cfg)`, covering `0..m` exactly.
    pub fn segments(&self, m: usize) -> Vec<(usize, usize, u8)> {
        let tbm = GEMM_CONFIGS[self.main as usize].tb.0 as usize;
        match self.tail {
            Some(t) if m > tbm && !m.is_multiple_of(tbm) => {
                let head = m / tbm * tbm;
                vec![(0, head, self.main), (head, m - head, t)]
            }
            _ => vec![(0, m, self.main)],
        }
    }
}

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

/// The launch plan for an `(M, N, K)` INT8 GEMM.
pub fn gemm_plan(m: usize, n: usize, k: usize) -> GemmPlan {
    GemmPlan::single(gemm_config(m, n, k))
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
                    let p = gemm_plan(m, n, k);
                    assert!((p.main as usize) < GEMM_CONFIGS.len());
                    assert!(p.tail.is_none_or(|t| (t as usize) < GEMM_CONFIGS.len()));
                }
            }
        }
    }

    #[test]
    fn segments_cover_every_row_once() {
        for m in [1usize, 2, 127, 128, 129, 4096, 4117, 4223, 8234] {
            for p in [
                GemmPlan::single(0),
                GemmPlan {
                    main: 0,
                    tail: Some(2),
                },
            ] {
                let segs = p.segments(m);
                let mut next = 0;
                for &(r0, rows, _) in &segs {
                    assert_eq!(r0, next);
                    assert!(rows > 0);
                    next += rows;
                }
                assert_eq!(next, m, "{p:?} m={m}");
            }
        }
        let s = GemmPlan {
            main: 0,
            tail: Some(2),
        }
        .segments(4117);
        assert_eq!(s, vec![(0, 4096, 0), (4096, 21, 2)]);
    }
}
