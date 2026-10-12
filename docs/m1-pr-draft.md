# Rascunho de PR — M1 (applegpu_g13) performance

> Enviado em 2026-10-12, dividido em dois PRs depois dos testes locais do autor:
> [jundot/omlx#4495](https://github.com/jundot/omlx/pull/4495) (correção do DFlash2) e
> [jundot/omlx#4496](https://github.com/jundot/omlx/pull/4496) (kernels M1, rascunho). O texto abaixo é o rascunho original.

## Summary

Speculative-verify, attention and DFlash fixes for M1-class GPUs (applegpu_g13), measured on
an M1 Max 64 GB with Qwen3.8-27B-oQ4e, Qwen3.6-35B-A3B-oQ4 and Gemma 4 26B-A4B-oQ4e. Every
new path is gated to generation 13 and has an `OMLX_*` switch to turn it off; other GPUs
keep their current routes.

- `patches/m1_verify_qmv.py`: multi-row qmv for 2-4 (4-bit) / 2-3 (5-bit) verify rows
  (+9..26% per projection).
- `patches/m1_verify_mma.py`: 5-16 row 4-bit verify matmul on simdgroup matrices with fp16
  inputs, adapted from mlx-dspark's `skinny_qmm` (MIT) (-8..26% per projection).
- `patches/qwen35_fa256_attention.py`: M1 route through query-chunked MLX SDPA for head_dim
  256 (from 3072 keys) and head_dim 512 (Gemma 4 global layers, from 32k keys), two blocks in
  flight and MLX buffer-cache trim; fp16 dispatch-budget calibration on M1.
- `patches/qwen35_verify_sdpa_split.py`: M1 key-split sizing for the tensor-op verify
  attention (-7..21%).
- `speculative/dflash_drafter.py`: DFlash2 drafters converted without `num_target_layers`
  load with the target's layer count.
- `engine_pool.py`: a drafter that fails to load logs an error stating speculation is off.

## Results (M1 Max 64 GB, tok/s)

See `docs/m1-performance-plan.md` ("Comparativo antes × depois") for the full tables. Highlights:
Qwen3.8-27B long-context decode +17..63% (DFlash2 with the new verify kernels),
Qwen3.6-35B-A3B long-context prefill +11..15%, Gemma 4 long-context prefill +17% at 103-136k.

## Test plan

- [ ] New tests: `tests/test_m1_verify_qmv.py`, `test_m1_verify_mma.py`, `test_fa256_m1_sdpa.py`,
      `test_m1_verify_attn_chunk.py`, `test_dflash2_num_target_layers.py`
- [ ] Existing suites unchanged versus `bafc6837` on the same runtime
- [x] Author's local end-to-end runs on M1 Max (agentic coding session with the three models, 2026-10-12)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
