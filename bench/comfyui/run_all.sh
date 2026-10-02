#!/usr/bin/env bash
# Full sequential suite: ours first, then every ComfyUI variant on a fresh server.
#   bash run_all.sh [variant ...]     # default: all
# Each variant: 4 prompts x 2 rounds (8 images), 1024^2, 40 steps, euler/simple, cfg 1.
# Results: ~/dev_tmp/comfy-bench/results/<variant>/
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
RV="bash $HERE/run_variant.sh"
BF=qwen_image_2.1_bf16.safetensors
I8=qwen_image_2.1_int8_convrot.safetensors
TE16=qwen3vl_8b_bf16.safetensors
TE8=qwen3vl_8b_int8_convrot.safetensors
TURBO=Qwen-Image-2.1-viggle-turbo-v0.3-6step-int8_convrot.safetensors
SIG='{"latent":"@latent","nodes":"1.0, 0.9375, 0.875, 0.75, 0.5, 0.25"}'

run() {
  local v=$1; shift
  echo "=== $v $(date +%T)"
  case $v in
    ours)        bash "$HERE/bench_ours.sh" ;;
    A_stock)     $RV $v ""                                      --unet $BF --clip $TE16 ;;
    T_template)  $RV $v ""                                      --unet $I8 --clip $TE8 ;;
    B_fp8_sage)  $RV $v "--use-sage-attention"                  --unet $BF --weight-dtype fp8_e4m3fn --clip $TE8 ;;
    C_fp8_sage_fast) $RV $v "--use-sage-attention --fast"       --unet $BF --weight-dtype fp8_e4m3fn --clip $TE8 ;;
    D_fp8_sage_fast_compile) $RV $v "--use-sage-attention --fast" --unet $BF --weight-dtype fp8_e4m3fn --clip $TE8 --compile ;;
    H1_int8_sage)    $RV $v "--use-sage-attention"              --unet $I8 --clip $TE8 ;;
    H2_int8_ck)      $RV $v "--use-ck-attention"                --unet $I8 --clip $TE8 ;;
    H3_int8_sage_fast) $RV $v "--use-sage-attention --fast"     --unet $I8 --clip $TE8 ;;
    H4_int8_ck_fast)   $RV $v "--use-ck-attention --fast"       --unet $I8 --clip $TE8 ;;
    H5_int8_${BEST_ATTN:-sage}_fast_compile) $RV $v "--use-${BEST_ATTN:-sage}-attention --fast" --unet $I8 --clip $TE8 --compile ;;
    # TorchCompileModel crashes under DynamicVRAM (comfy_aimdo malloc_graph is traced by dynamo); retry with it off
    H6_int8_sage_fast_nodynvram)  $RV $v "--use-sage-attention --fast --disable-dynamic-vram" --unet $I8 --clip $TE8 ;;
    H7_int8_sage_fast_nodynvram_compile) $RV $v "--use-sage-attention --fast --disable-dynamic-vram" --unet $I8 --clip $TE8 --compile ;;
    D2_fp8_sage_fast_nodynvram_compile) $RV $v "--use-sage-attention --fast --disable-dynamic-vram" --unet $BF --weight-dtype fp8_e4m3fn --clip $TE8 --compile ;;
    G_turbo6)    $RV $v "--use-${BEST_ATTN:-sage}-attention --fast" --unet $TURBO --clip $TE8 \
                   --sigmas-node ViggleTurboSigmas --sigmas "$SIG" --steps 6 ;;
    *) echo "unknown variant $v" ;;
  esac
  echo "=== END $v rc=$? $(date +%T)"
  sleep 20   # let clocks/temps settle between variants
}

VARIANTS=("$@")
[ ${#VARIANTS[@]} -eq 0 ] && VARIANTS=(ours A_stock T_template B_fp8_sage C_fp8_sage_fast D_fp8_sage_fast_compile \
  H1_int8_sage H2_int8_ck H3_int8_sage_fast H4_int8_ck_fast H5_int8_sage_fast_compile \
  H6_int8_sage_fast_nodynvram H7_int8_sage_fast_nodynvram_compile D2_fp8_sage_fast_nodynvram_compile G_turbo6)
for v in "${VARIANTS[@]}"; do run "$v"; done
echo SUITE_END
