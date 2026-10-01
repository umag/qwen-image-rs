//! Tile configs and launch plans of the ConvRot INT8 CUTLASS EVT GEMM
//! (`qwen-image-rs-gemm-merge-tune`, `qwen-image-rs-gemm-tiling-push`).
//! CPU-compiled so the selection is unit-tested on every build; the kernels
//! live in `kernels/convrot/int8_gemm.cu` (`Cfg<i>`, same order).
//!
//! INT8 accumulation is exact and the dequant epilogue is per element, so
//! every config and every row split computes bit-identical outputs: the
//! choice is speed only, and it is a fixed function of the shape
//! (deterministic, no split-K, no stream-K).

/// One CUTLASS tile config: threadblock and warp tile, pipeline stages and
/// threadblock swizzle (CTAs rastered in groups of `swizzle` N-tiles).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TileConfig {
    pub tb: (u16, u16, u16),
    pub warp: (u16, u16, u16),
    pub stages: u8,
    pub swizzle: u8,
}

const fn cfg(tb: (u16, u16, u16), warp: (u16, u16, u16), stages: u8, swizzle: u8) -> TileConfig {
    TileConfig {
        tb,
        warp,
        stages,
        swizzle,
    }
}

/// The compiled configs, by index (= the `cfg` passed to the launcher).
pub const GEMM_CONFIGS: [TileConfig; 5] = [
    // 0: the original config; kept for shapes outside the measured domain.
    cfg((128, 128, 64), (64, 64, 64), 3, 1),
    // 1: swizzled 128x128: merged q|k (N = 8192) at every B.
    cfg((128, 128, 64), (64, 64, 64), 3, 4),
    // 2: small tiles: tail linears (M = 2 / txt, N = 64) and split tails.
    cfg((64, 64, 64), (32, 32, 64), 6, 1),
    // 3: 128x256 swizzled: N = 4096 heads, and N >= 16384 at B >= 2.
    cfg((128, 256, 64), (64, 64, 64), 3, 4),
    // 4: 128x256 4-stage: N >= 16384 (merged gate|proj) at B = 1.
    cfg((128, 256, 64), (64, 64, 64), 4, 1),
];

/// How one GEMM is launched. `main` covers the rows up to the last full
/// `main` M-tile; with `tail = Some(t)` the remaining `M mod TBM` rows run
/// as a second launch with config `t`. At M = txt + 4096 the text rows past
/// the 4096 image rows otherwise add a near-empty last M-tile row of CTAs,
/// i.e. one extra partial wave (N = 4096: 33·32 = 1056 CTAs = 4.125 waves).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GemmPlan {
    pub main: u8,
    pub tail: Option<u8>,
}

impl GemmPlan {
    pub const fn single(main: u8) -> Self {
        Self { main, tail: None }
    }

    /// The launches as `(first_row, rows, cfg)`, covering `0..m` exactly
    /// once. A tail segment starts at a multiple of the main config's TBM.
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

/// At or below this M (or N) a GEMM is "small" (tail linears at M = 2 or
/// the text length, proj_out at N = 64).
const SMALL: usize = 64;
/// The measured domain of the tuned plans: one 1024² lane (4096 image rows)
/// and more. Below it the original config stays.
const TUNED_M: usize = 4096;
/// From this M on (B >= 2 at 1024²) swizzle 4 wins for the widest GEMMs.
const LARGE_M: usize = 8192;

/// The launch plan for an `(M, N, K)` INT8 GEMM, from `gemm-bench` (RTX
/// 4090, min of 3 interleaved rounds at txt = 21 / 60 / 120 and B = 1 / 2;
/// HANDOVER "GEMM tiling push"):
/// - small M or N -> 64x64 6-stage;
/// - N >= 16384 (gate|proj) -> 128x256, 4-stage at B = 1, swizzle 4 at
///   B >= 2; no split (the tail launch would re-read the 100 MB weight);
/// - N >= 8192 (q|k) -> 128x128 swizzle 4 (every config within ~1%);
/// - N <= 4096-class (to_v / to_out / MLP out, K = 4096 and 12288) ->
///   128x256 swizzle 4 over the full 128-row tiles + the < 128 tail rows
///   on 64x64 (-5..-8%: exact waves for the head).
pub fn gemm_plan(m: usize, n: usize, _k: usize) -> GemmPlan {
    if m <= SMALL || n <= SMALL {
        GemmPlan::single(2)
    } else if m < TUNED_M {
        GemmPlan::single(0)
    } else if n >= 16384 {
        GemmPlan::single(if m >= LARGE_M { 3 } else { 4 })
    } else if n >= 8192 {
        GemmPlan::single(1)
    } else {
        GemmPlan {
            main: 3,
            tail: Some(2),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SPLIT: GemmPlan = GemmPlan {
        main: 3,
        tail: Some(2),
    };

    #[test]
    fn b1_dit_shapes() {
        for txt in [5usize, 21, 60, 120] {
            let m = 4096 + txt;
            assert_eq!(gemm_plan(m, 4096, 4096), SPLIT, "to_v / to_out");
            assert_eq!(gemm_plan(m, 8192, 4096), GemmPlan::single(1), "q|k");
            assert_eq!(gemm_plan(m, 24576, 4096), GemmPlan::single(4), "gate|proj");
            assert_eq!(gemm_plan(m, 4096, 12288), SPLIT, "mlp out");
        }
    }

    #[test]
    fn batched_dit_shapes() {
        for m in [8234usize, 16468] {
            assert_eq!(gemm_plan(m, 4096, 4096), SPLIT);
            assert_eq!(gemm_plan(m, 8192, 4096), GemmPlan::single(1));
            assert_eq!(gemm_plan(m, 24576, 4096), GemmPlan::single(3));
            assert_eq!(gemm_plan(m, 4096, 12288), SPLIT);
        }
    }

    #[test]
    fn tails_take_the_small_tile() {
        assert_eq!(gemm_plan(2, 16384, 4096), GemmPlan::single(2)); // modulation
        assert_eq!(gemm_plan(2, 4096, 256), GemmPlan::single(2)); // time_embed.linear_1
        assert_eq!(gemm_plan(37, 4096, 4096), GemmPlan::single(2)); // txt_in
        assert_eq!(gemm_plan(4117, 64, 4096), GemmPlan::single(2)); // proj_out
    }

    #[test]
    fn untuned_mid_m_keeps_the_original_config() {
        for m in [65usize, 1024, 1041, 4095] {
            assert_eq!(gemm_plan(m, 4096, 4096), GemmPlan::single(0));
        }
    }

    #[test]
    fn every_selected_index_is_compiled() {
        for m in [1usize, 2, 64, 65, 4117, 8192, 16468] {
            for n in [64usize, 4096, 8192, 24576] {
                for k in [256usize, 4096, 12288] {
                    let p = gemm_plan(m, n, k);
                    assert!((p.main as usize) < GEMM_CONFIGS.len());
                    assert!(p.tail.is_none_or(|t| (t as usize) < GEMM_CONFIGS.len()));
                }
            }
        }
    }

    #[test]
    fn segments_cover_every_row_once_on_tile_boundaries() {
        for m in [1usize, 2, 127, 128, 129, 4096, 4117, 4160, 4223, 8234] {
            for p in [GemmPlan::single(0), SPLIT] {
                let segs = p.segments(m);
                let tbm = GEMM_CONFIGS[p.main as usize].tb.0 as usize;
                let mut next = 0;
                for (i, &(r0, rows, _)) in segs.iter().enumerate() {
                    assert_eq!(r0, next);
                    assert!(rows > 0);
                    if i > 0 {
                        assert!(r0.is_multiple_of(tbm), "tail start {r0} not on a tile edge");
                        assert!(rows < tbm);
                    }
                    next += rows;
                }
                assert_eq!(next, m, "{p:?} m={m}");
            }
        }
        assert_eq!(SPLIT.segments(4117), vec![(0, 4096, 3), (4096, 21, 2)]);
        assert_eq!(SPLIT.segments(4096), vec![(0, 4096, 3)]);
    }

    #[test]
    fn split_main_tiles_divide_the_image_rows() {
        // A 1024² lane has 4096 image rows: the split main config's TBM must
        // divide it so the head launch has no partial M-tile.
        assert!(4096usize.is_multiple_of(GEMM_CONFIGS[SPLIT.main as usize].tb.0 as usize));
    }
}
