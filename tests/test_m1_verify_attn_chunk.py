# SPDX-License-Identifier: Apache-2.0
"""M1 key-split sizing for the tensor-op verify attention."""

from __future__ import annotations

from omlx.patches import qwen35_verify_sdpa_split as vs


def test_about_one_threadgroup_per_core_within_bounds():
    # M1 Max (32 cores), Qwen3.8 dense layout (4 KV heads)
    assert vs._m1_gqa_chunk(32768, 4, cores=32) == 4096
    assert vs._m1_gqa_chunk(131072, 4, cores=32) == 4096
    # Qwen3.6 MoE layout (2 KV heads)
    assert vs._m1_gqa_chunk(32768, 2, cores=32) == 2048
    assert vs._m1_gqa_chunk(131072, 2, cores=32) == 4096
    # never below 1024 keys, always a multiple of 64
    assert vs._m1_gqa_chunk(17000, 1, cores=32) == 1024
    for cores in (14, 16, 24, 32, 48, 64):
        for kv_len in (16385, 50000, 99999):
            for heads in (1, 2, 4, 8):
                chunk = vs._m1_gqa_chunk(kv_len, heads, cores=cores)
                assert 1024 <= chunk <= 4096 and chunk % 64 == 0


def test_more_cores_give_more_splits():
    # M1 Ultra (64 cores) needs twice the splits of an M1 Max for the same context.
    kv_len, kv_heads = 27063, 2
    max_splits = -(-kv_len // vs._m1_gqa_chunk(kv_len, kv_heads, cores=32))
    ultra_splits = -(-kv_len // vs._m1_gqa_chunk(kv_len, kv_heads, cores=64))
    assert max_splits == 16
    assert ultra_splits == 27


def test_core_count_comes_from_the_hardware_with_an_m1_max_fallback(monkeypatch):
    from omlx.utils import hardware

    vs._m1_gpu_cores.cache_clear()
    monkeypatch.setattr(hardware, "get_gpu_core_count", lambda: 64)
    assert vs._m1_gqa_chunk(27063, 2) == vs._m1_gqa_chunk(27063, 2, cores=64)
    vs._m1_gpu_cores.cache_clear()
    monkeypatch.setattr(hardware, "get_gpu_core_count", lambda: None)
    assert vs._m1_gpu_cores() == 32
    assert vs._m1_gqa_chunk(27063, 2) == vs._m1_gqa_chunk(27063, 2, cores=32)
    vs._m1_gpu_cores.cache_clear()
