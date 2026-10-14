# SPDX-License-Identifier: Apache-2.0
"""DSpark drafters in the batched DFlash round loop.

mlx-vlm loads DSpark checkpoints as draft kind "dflash". Their proposals differ
from DFlash in two ways the batched loop has to reproduce: with
``sample_from_anchor`` the anchor slot yields the first proposal, and the
Markov head corrects each position from the previously chosen token.
"""

from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.sample_utils import make_sampler
from mlx_vlm.speculative.drafters.dspark.config import DSparkConfig
from mlx_vlm.speculative.drafters.dspark.dspark import DSparkDraftModel

from omlx.speculative import dflash_drafter as dd

VOCAB = 64
HIDDEN = 32
TARGET_LAYERS = 6
TARGET_LAYER_IDS = [1, 3, 5]
WINDOW = 12
PROPOSALS = 3  # checkpoint block size; mlx-vlm verifies anchor + proposals


def _tiny_config():
    return DSparkConfig.from_dict(
        {
            "model_type": "qwen3",
            "architectures": ["DSparkDraftModel"],
            "hidden_size": HIDDEN,
            "intermediate_size": 48,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "vocab_size": VOCAB,
            "rms_norm_eps": 1e-6,
            "max_position_embeddings": 4096,
            "num_target_layers": TARGET_LAYERS,
            "sliding_window": WINDOW,
            "layer_types": ["sliding_attention", "sliding_attention"],
            "rope_parameters": {"rope_theta": 10000.0, "rope_type": "default"},
            "tie_word_embeddings": False,
            "block_size": PROPOSALS,
            "dflash_config": {
                "markov_rank": 8,
                "markov_head_type": "vanilla",
                "mask_token_id": VOCAB - 1,
                "projector_type": "dspark",
                "target_layer_ids": TARGET_LAYER_IDS,
            },
        }
    )


def _tiny_target():
    embed = nn.Embedding(VOCAB, HIDDEN)
    lm_head = nn.Linear(HIDDEN, VOCAB, bias=False)
    language = SimpleNamespace(
        config={
            "text_config": {
                "hidden_size": HIDDEN,
                "num_hidden_layers": TARGET_LAYERS,
                "vocab_size": VOCAB,
            }
        },
        model=SimpleNamespace(layers=[object()] * TARGET_LAYERS, embed_tokens=embed),
        lm_head=lm_head,
        rollback_speculative_cache=lambda *args, **kwargs: None,
    )
    mx.eval(embed.parameters(), lm_head.parameters())
    return SimpleNamespace(language_model=language)


def _tiny_drafter(seed=0):
    mx.random.seed(seed)
    model = DSparkDraftModel(_tiny_config())
    params = {
        key: mx.random.normal(value.shape) * 0.2
        for key, value in nn.utils.tree_flatten(model.parameters())
    }
    model.load_weights(list(params.items()), strict=False)
    model.bind(_tiny_target())
    mx.eval(model.parameters())
    block = dd.resolve_block_size(model, None)
    return dd.DFlashDrafter(model, block_size=block, source_path="tiny")


def _context(n, seed):
    mx.random.seed(seed)
    return mx.concatenate(
        [mx.random.normal((1, n, HIDDEN)) for _ in TARGET_LAYER_IDS], axis=-1
    )


def test_block_covers_anchor_and_proposals():
    drafter = _tiny_drafter()
    assert drafter.block_size == PROPOSALS + 1
    assert dd._dspark_anchor_slot(drafter.model)


def test_batched_forward_matches_reference_draft_block():
    """Anchor-slot block layout and Markov transitions match mlx-vlm."""
    for length in (1, 3, WINDOW - 1):
        drafter = _tiny_drafter()
        context = _context(length, seed=60 + length)
        anchor = mx.array([7], dtype=mx.int32)
        cache = drafter.model.make_cache()
        for layer_cache in cache:
            layer_cache.offset = 0
        expected = drafter.model.draft_block(
            anchor,
            context,
            cache,
            drafter.block_size,
            lambda logits: mx.argmax(logits, axis=-1),
        )
        row = drafter._row(0)
        got = drafter._draft_batched(
            [(SimpleNamespace(uid=0), row, context, anchor, None)]
        )[0][0]
        assert got.tolist() == expected.tolist()


def test_select_markov_greedy_matches_sample_block():
    mx.random.seed(3)
    markov = _tiny_drafter().model.markov_head
    logits = mx.random.normal((2, PROPOSALS, VOCAB))
    anchors = mx.array([5, 9], dtype=mx.int32)
    expected = markov.sample_block(
        logits,
        first_prev_tokens=anchors,
        sampler=lambda x: mx.argmax(x, axis=-1),
    )
    got = dd._select_markov(markov, logits, anchors, [None, None])
    assert [tokens.tolist() for tokens, _ in got] == [
        expected[i : i + 1].tolist() for i in range(2)
    ]


def test_select_markov_sampled_rows_keep_proposal_logprobs():
    mx.random.seed(4)
    markov = _tiny_drafter().model.markov_head
    logits = mx.random.normal((2, PROPOSALS, VOCAB))
    anchors = mx.array([5, 9], dtype=mx.int32)
    sampler = make_sampler(temp=0.7, top_p=0.9)
    (greedy, greedy_lps), (sampled, sampled_lps) = dd._select_markov(
        markov, logits, anchors, [None, sampler]
    )
    assert greedy.shape == (1, PROPOSALS) and greedy_lps == []
    assert sampled.shape == (1, PROPOSALS)
    assert len(sampled_lps) == PROPOSALS
    assert all(lp.shape == (VOCAB,) for lp in sampled_lps)
