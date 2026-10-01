//! Builds the ConvRot INT8 CUDA kernel under the `convrot` feature.
//! Needs a fresh CUTLASS: set `CUTLASS_DIR` to its `include/` dir (the
//! flash-attn-vendored CUTLASS has a `matrix.h` bug CUDA 13.3 rejects).

fn main() {
    println!("cargo:rerun-if-changed=kernels/convrot/int8_gemm.cu");
    println!("cargo:rerun-if-changed=kernels/convrot/quant_ops.cu");
    println!("cargo:rerun-if-env-changed=CUTLASS_DIR");

    #[cfg(feature = "convrot")]
    {
        let cutlass = std::env::var("CUTLASS_DIR").unwrap_or_else(|_| {
            format!(
                "{}/dev_tmp/cutlass/include",
                std::env::var("HOME").unwrap_or_default()
            )
        });
        let cap = std::env::var("CUDA_COMPUTE_CAP").unwrap_or_else(|_| "89".into());
        cc::Build::new()
            .cuda(true)
            .flag("-std=c++17")
            .flag("--expt-relaxed-constexpr")
            .flag("--expt-extended-lambda") // EVT epilogue visitors use device lambdas
            .flag(format!("-arch=sm_{cap}"))
            .include(&cutlass)
            .file("kernels/convrot/int8_gemm.cu")
            .file("kernels/convrot/quant_ops.cu")
            .compile("convrot_int8");
        println!("cargo:rustc-link-lib=dylib=cudart");
    }

    #[cfg(feature = "fusednorm")]
    {
        // Fused LayerNorm+AdaLN kernel (no external deps).
        println!("cargo:rerun-if-changed=kernels/fusednorm/fused_norm.cu");
        println!("cargo:rerun-if-changed=kernels/fusednorm/head_rmsnorm.cuh");
        let cap = std::env::var("CUDA_COMPUTE_CAP").unwrap_or_else(|_| "89".into());
        cc::Build::new()
            .cuda(true)
            .flag("-std=c++17")
            .flag("--expt-relaxed-constexpr")
            .flag(format!("-arch=sm_{cap}"))
            .file("kernels/fusednorm/fused_norm.cu")
            .compile("fused_norm");
        println!("cargo:rustc-link-lib=dylib=cudart");
    }

    #[cfg(feature = "sage")]
    {
        // Vendored SageAttention INT8-QK / FP16-PV kernel (thu-ml), torch-free,
        // plus the BSHD interleaved-RoPE kernel used only by the sage path.
        println!("cargo:rerun-if-changed=kernels/sage/sage_ffi.cu");
        println!("cargo:rerun-if-changed=kernels/sage/rope_bshd.cu");
        println!("cargo:rerun-if-changed=kernels/sage/rope_pair.cuh");
        let cap = std::env::var("CUDA_COMPUTE_CAP").unwrap_or_else(|_| "89".into());
        cc::Build::new()
            .cuda(true)
            .flag("-std=c++17")
            .flag("--expt-relaxed-constexpr")
            .flag("--expt-extended-lambda")
            .flag(format!("-arch=sm_{cap}"))
            .flag("-diag-suppress=177") // unused CHECK_ macros in vendored utils
            .include("kernels/sage")
            .file("kernels/sage/sage_ffi.cu")
            .file("kernels/sage/rope_bshd.cu")
            .compile("sage_attn");
        println!("cargo:rustc-link-lib=dylib=cudart");
    }

    #[cfg(feature = "sage2")]
    {
        // SageAttention2 sm89 (INT8-QK per-thread + FP8-PV): its own TU so the
        // vendored sm80/sm89 kernels' PACK_SIZE_* macros never meet.
        println!("cargo:rerun-if-changed=kernels/sage/sage2_ffi.cu");
        println!("cargo:rerun-if-changed=kernels/fusednorm/head_rmsnorm.cuh");
        println!("cargo:rerun-if-changed=kernels/sage/vendor/qattn/qk_int_sv_f8_sm89.cuh");
        let cap = std::env::var("CUDA_COMPUTE_CAP").unwrap_or_else(|_| "89".into());
        cc::Build::new()
            .cuda(true)
            .flag("-std=c++17")
            .flag("--expt-relaxed-constexpr")
            .flag("--expt-extended-lambda")
            .flag(format!("-arch=sm_{cap}"))
            .flag("-diag-suppress=177")
            .include("kernels/sage")
            .file("kernels/sage/sage2_ffi.cu")
            .compile("sage2_attn");
        println!("cargo:rustc-link-lib=dylib=cudart");
    }
}
