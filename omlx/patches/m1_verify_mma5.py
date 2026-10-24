# SPDX-License-Identifier: Apache-2.0
"""Verify-width (5-16 rows) 5-bit quantized matmul on simdgroup matrices, M1.

The 5-bit counterpart of ``m1_verify_mma`` (same layout: weights as MMA
operand A loaded straight into registers, up to two 8-row tiles of X per
weight read, scale/bias factored out per 64-input group, split-K over the
threadgroup's simdgroups). 5-bit packing puts 8 values in 5 bytes, so the
8-value chunk k = 8j..8j+7 of a group is bytes 5j..5j+4. A lane's two chunks
(j = fn, fn + 1, fn even) are 10 bytes at an even offset, read as three
aligned words and split into a 32-bit low word and an 8-bit high byte per
chunk. MMA step s takes field s: fields 0..5 masked in place in the low word
against inputs pre-scaled by 2^-5s, field 7 masked in place in the high byte
against 2^-3, and field 6, which straddles the two, shifted out (pre-scale 1).

Weight placement: reading 8 weight rows at once, this kernel is sensitive to
how the weight buffers are mapped. On weights as loaded (one buffer per
safetensors shard), Metal GPU counters show a 15-20% TLB miss rate and the
kernel runs 2-3x slower than sg8; on the same weights copied into their own
buffers the miss rate is ~2.5% and it is faster. So the first time a
projection takes this route its weight, scales and biases are replaced by
copies in fresh buffers (bit-identical; the originals are released). The
one-row decode and the other routes measured the same on both placements.

Numerics: fp32 accumulation; row-count invariant; not bit-identical to the
one-row decode (same class as the other verify kernels).

Measured on M1 Max (fp16, Qwen3.8-27B 5-bit projections, chain over all
layers, vs the sg8 / m1 route they replace):

    rows             5     8     9    12    16
    mlp.down_proj  -10%  -12%  -21%  -22%  -21%
    gdn.in_proj_z   -1%   -8%  -21%  -21%  -19%
    gdn.out_proj   -12%  -23%  -41%  -41%  -39%

Gating: applegpu_g13 only (where it was measured).
"""

from __future__ import annotations

import logging
import math
from functools import cache

from . import m1_verify_mma

logger = logging.getLogger(__name__)

GROUP = 64
BITS = 5
MIN_ROWS = 5
MAX_ROWS = 16
_MIN_N = 1024

_SRC = r"""
  constexpr int G = K / 64;                   // quant groups along K
  constexpr int FIELDS = 8;                   // 5-bit fields per chunk = MMA steps
  constexpr int UPR = K * 5 / 16;             // packed ushorts per weight row
  const int lane = int(thread_index_in_simdgroup);
  const int c = int(simdgroup_index_in_threadgroup);
  const int fm = ((lane >> 4) << 2) + ((lane >> 1) & 3);       // fragment row
  const int fn = (((lane >> 3) & 1) << 2) + ((lane & 1) << 1);  // fragment col pair
  const int R = X_shape[0];
  const int n0 = int(threadgroup_position_in_grid.x) * (8 * NT);
  const int m0 = int(threadgroup_position_in_grid.y) * (8 * RT);

  int nrow[NT];
  for (int t = 0; t < NT; ++t) nrow[t] = min(n0 + 8 * t + fm, N - 1);
  const device T* x0[RT];
  const device T* x1[RT];
  for (int r = 0; r < RT; ++r) {
    x0[r] = X + size_t(min(m0 + 8 * r + fn, R - 1)) * K;
    x1[r] = X + size_t(min(m0 + 8 * r + fn + 1, R - 1)) * K;
  }
  float acc[RT][NT][2];
  for (int r = 0; r < RT; ++r)
    for (int t = 0; t < NT; ++t) { acc[r][t][0] = 0.0f; acc[r][t][1] = 0.0f; }

  for (int g = c; g < G; g += S) {
    simdgroup_matrix<float, 8, 8> P[RT][NT];
    float sum0[RT], sum1[RT];
    for (int r = 0; r < RT; ++r) {
      sum0[r] = 0.0f; sum1[r] = 0.0f;
      for (int t = 0; t < NT; ++t) P[r][t] = simdgroup_matrix<float, 8, 8>(0.0f);
    }
    // Chunks fn and fn + 1 of group g (10 bytes at byte 5 * fn of the
    // group's 40) from three aligned words. The group starts 8-byte aligned;
    // for fn = 2 and 6 the run starts 2 bytes into its first word.
    uint lo0[NT], hi0[NT], lo1[NT], hi1[NT];
    const int wfirst = (5 * fn) / 4;            // 0, 2, 5, 7
    const int sh = ((5 * fn) % 4) * 8;          // 0 or 16 bits
    for (int t = 0; t < NT; ++t) {
      const device uint* p = (const device uint*)Wq + size_t(nrow[t]) * (UPR / 2) + 10 * g + wfirst;
      const uint w0 = p[0], w1 = p[1], w2 = p[2];
      // The 10 bytes as b[0..9]: a = bytes 0..3, b = bytes 4..7, c = bytes 8..9.
      const uint a = sh ? ((w0 >> 16) | (w1 << 16)) : w0;
      const uint b = sh ? ((w1 >> 16) | (w2 << 16)) : w1;
      const uint c = sh ? (w2 >> 16) : (w2 & 0xFFFFu);
      lo0[t] = a;
      hi0[t] = b & 0xFFu;
      lo1[t] = (b >> 8) | (c << 24);
      hi1[t] = c >> 8;
    }
    const int xo = 64 * g + FIELDS * fm;      // this lane's 8-input run
    uint4 xa[RT], xb[RT];
    for (int r = 0; r < RT; ++r) {
      xa[r] = *(const device uint4*)(x0[r] + xo);
      xb[r] = *(const device uint4*)(x1[r] + xo);
      for (int e = 0; e < FIELDS; ++e) {
        sum0[r] += x_at<T>(xa[r][e / 2], e);
        sum1[r] += x_at<T>(xb[r][e / 2], e);
      }
    }
    #pragma clang loop unroll(full)
    for (int s = 0; s < FIELDS; ++s) {
      // 2^-(5s) for the in-place fields 0..5, 1 for the shifted field 6,
      // 2^-3 for field 7 (bits 3..7 of the high byte).
      const float pre = s < 6 ? as_type<float>(uint(127 - 5 * s) << 23)
                              : (s == 6 ? 1.0f : 0.125f);
      simdgroup_matrix<float, 8, 8> b[RT];
      for (int r = 0; r < RT; ++r) {
        b[r].thread_elements()[0] = x_at<T>(xa[r][s / 2], s) * pre;
        b[r].thread_elements()[1] = x_at<T>(xb[r][s / 2], s) * pre;
      }
      for (int t = 0; t < NT; ++t) {
        simdgroup_matrix<float, 8, 8> a;
        if (s < 6) {
          const uint mask = 0x1Fu << (5 * s);
          a.thread_elements()[0] = float(lo0[t] & mask);
          a.thread_elements()[1] = float(lo1[t] & mask);
        } else if (s == 6) {
          a.thread_elements()[0] = float((lo0[t] >> 30) | ((hi0[t] & 7u) << 2));
          a.thread_elements()[1] = float((lo1[t] >> 30) | ((hi1[t] & 7u) << 2));
        } else {
          a.thread_elements()[0] = float(hi0[t] & 0xF8u);
          a.thread_elements()[1] = float(hi1[t] & 0xF8u);
        }
        for (int r = 0; r < RT; ++r) simdgroup_multiply_accumulate(P[r][t], a, b[r], P[r][t]);
      }
    }
    for (int r = 0; r < RT; ++r) {            // group input sums over the 8 fm lanes
      for (int d = 2; d <= 16; d <<= 1) {
        if (d == 8) continue;
        sum0[r] += simd_shuffle_xor(sum0[r], ushort(d));
        sum1[r] += simd_shuffle_xor(sum1[r], ushort(d));
      }
    }
    for (int t = 0; t < NT; ++t) {
      const float sc = float(SC[size_t(nrow[t]) * G + g]);
      const float bi = float(BI[size_t(nrow[t]) * G + g]);
      for (int r = 0; r < RT; ++r) {
        acc[r][t][0] = fma(bi, sum0[r], fma(sc, P[r][t].thread_elements()[0], acc[r][t][0]));
        acc[r][t][1] = fma(bi, sum1[r], fma(sc, P[r][t].thread_elements()[1], acc[r][t][1]));
      }
    }
  }

  threadgroup float part[S * RT * NT * 64];   // split-K partials
  for (int r = 0; r < RT; ++r)
    for (int t = 0; t < NT; ++t)
      for (int e = 0; e < 2; ++e) part[((c * RT + r) * NT + t) * 64 + fm * 8 + fn + e] = acc[r][t][e];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int i = c * 32 + lane; i < RT * NT * 64; i += S * 32) {
    float v = 0.0f;
    for (int q = 0; q < S; ++q) v += part[q * (RT * NT * 64) + i];
    const int r = i / (NT * 64), t = (i / 64) % NT, e = i % 64;
    const int row = m0 + 8 * r + (e % 8), n = n0 + 8 * t + (e / 8);
    if (row < R && n < N) OUT[size_t(row) * N + n] = T(v);
  }
"""




def enabled() -> bool:
    """Same gate as the 4-bit kernel (applegpu_g13)."""
    return m1_verify_mma.enabled()


def eligible(M: int, K: int, N: int, bits: int, group_size: int, dtype) -> bool:
    import mlx.core as mx

    if int(bits) != BITS or not enabled():
        return False
    return (
        MIN_ROWS <= int(M) <= MAX_ROWS
        and int(group_size) == GROUP
        and dtype in (mx.float16, mx.bfloat16)
        and int(K) % GROUP == 0
        and int(N) % 8 == 0
        and int(N) >= _MIN_N
    )


def own_buffers(layer) -> None:
    """Move a QuantizedLinear's weight, scales and biases into fresh buffers, once."""
    if getattr(layer, "_omlx_mma5_own_buffers", False):
        return
    import mlx.core as mx

    # Elementwise no-ops produce new arrays, so each lands in its own buffer.
    layer.weight = layer.weight ^ mx.array(0, dtype=layer.weight.dtype)
    layer.scales = layer.scales * mx.array(1, dtype=layer.scales.dtype)
    layer.biases = layer.biases * mx.array(1, dtype=layer.biases.dtype)
    mx.eval(layer.weight, layer.scales, layer.biases)
    layer._omlx_mma5_own_buffers = True
    logger.debug("m1_mma5: moved a %s projection into its own buffers", tuple(layer.weight.shape))


@cache
def _kernel():
    import mlx.core as mx

    return mx.fast.metal_kernel(
        name="omlx_m1_verify_mma5",
        input_names=["X", "Wq", "SC", "BI"],
        output_names=["OUT"],
        source=_SRC,
        header=m1_verify_mma._HEADER,
    )


def m1_mma5_qmm(x2, w_q, scales, biases):
    """(M, K) x (N, K)^T -> (M, N), 5-bit affine groups of 64, M in 5..16."""
    rows, k = int(x2.shape[0]), int(x2.shape[1])
    n = int(w_q.shape[0])
    rt = min(m1_verify_mma.RT_MAX, (rows + 7) // 8)
    nt, s = m1_verify_mma.geometry(n, rt)
    (y,) = _kernel()(
        inputs=[x2, w_q, scales, biases],
        template=[("T", x2.dtype), ("K", k), ("N", n), ("NT", nt), ("RT", rt), ("S", s)],
        grid=(math.ceil(n / (8 * nt)) * s * 32, math.ceil(rows / (8 * rt)), 1),
        threadgroup=(s * 32, 1, 1),
        output_shapes=[(rows, n)],
        output_dtypes=[x2.dtype],
    )
    return y
