//! Builds the ConvRot INT8 CUDA kernel under the `convrot` feature.
//! Needs a fresh CUTLASS: set `CUTLASS_DIR` to its `include/` dir (the
//! flash-attn-vendored CUTLASS has a `matrix.h` bug CUDA 13.3 rejects).

fn main() {
    println!("cargo:rerun-if-changed=kernels/convrot/int8_gemm.cu");
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
            .flag(format!("-arch=sm_{cap}"))
            .include(&cutlass)
            .file("kernels/convrot/int8_gemm.cu")
            .compile("convrot_int8");
        println!("cargo:rustc-link-lib=dylib=cudart");
    }
}
