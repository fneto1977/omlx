# SPDX-License-Identifier: Apache-2.0
"""M1 key-split sizing for the tensor-op verify attention."""

from __future__ import annotations

from omlx.patches import qwen35_verify_sdpa_split as vs


def test_about_one_threadgroup_per_core_within_bounds():
    # Qwen3.8 dense layout (4 KV heads)
    assert vs._m1_gqa_chunk(32768, 4) == 4096
    assert vs._m1_gqa_chunk(131072, 4) == 4096
    # Qwen3.6 MoE layout (2 KV heads)
    assert vs._m1_gqa_chunk(32768, 2) == 2048
    assert vs._m1_gqa_chunk(131072, 2) == 4096
    # never below 1024 keys, always a multiple of 64
    assert vs._m1_gqa_chunk(17000, 1) == 1024
    for kv_len in (16385, 50000, 99999):
        for heads in (1, 2, 4, 8):
            chunk = vs._m1_gqa_chunk(kv_len, heads)
            assert 1024 <= chunk <= 4096 and chunk % 64 == 0
