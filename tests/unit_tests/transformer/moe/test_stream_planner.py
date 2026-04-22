# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest
import torch

from megatron.core.transformer.moe.stream_planner import build_stream_round_plan
from megatron.core.transformer.transformer_config import TransformerConfig


def _make_base_stream_config(**kwargs) -> TransformerConfig:
    defaults = dict(
        num_layers=1,
        hidden_size=16,
        num_attention_heads=4,
        num_moe_experts=4,
        moe_router_topk=2,
        moe_token_dispatcher_type="stream",
        moe_router_load_balancing_type="aux_loss",
        use_cpu_initialization=True,
        add_bias_linear=False,
        recompute_granularity="selective",
        recompute_modules=["moe"],
    )
    defaults.update(kwargs)
    return TransformerConfig(**defaults)


def test_stream_round_plan_covers_all_tokens_for_all_rounds():
    topk_indices = torch.tensor(
        [
            [3, 1],
            [0, 2],
            [1, 3],
        ],
        dtype=torch.long,
    )
    topk_probs = torch.tensor(
        [
            [0.8, 0.2],
            [0.6, 0.4],
            [0.7, 0.3],
        ],
        dtype=torch.float32,
    )

    plan = build_stream_round_plan(topk_probs, topk_indices)

    assert plan.num_rounds == 2
    assert plan.num_tokens == 3
    assert all(round_assignment.num_tokens == 3 for round_assignment in plan.rounds)
    reconstructed_indices = torch.stack(
        [round_assignment.expert_indices for round_assignment in plan.rounds], dim=1
    )
    reconstructed_probs = torch.stack(
        [round_assignment.probs for round_assignment in plan.rounds], dim=1
    )
    torch.testing.assert_close(reconstructed_indices, topk_indices)
    torch.testing.assert_close(reconstructed_probs, topk_probs)


def test_stream_round_plan_rejects_shape_mismatch():
    topk_indices = torch.tensor([[0, 1], [2, 3]], dtype=torch.long)
    topk_probs = torch.tensor([[0.5, 0.5]], dtype=torch.float32)

    with pytest.raises(ValueError, match="same shape"):
        build_stream_round_plan(topk_probs, topk_indices)


def test_stream_config_requires_moe_recompute():
    with pytest.raises(ValueError, match="requires MoE recompute"):
        _make_base_stream_config(recompute_granularity=None, recompute_modules=[])


def test_stream_config_rejects_shared_experts():
    with pytest.raises(ValueError, match="does not support shared experts"):
        _make_base_stream_config(moe_shared_expert_intermediate_size=32)


def test_stream_config_rejects_moe_capacity():
    with pytest.raises(ValueError, match="does not support token dropping or expert capacity"):
        _make_base_stream_config(moe_expert_capacity_factor=1.0)


def test_stream_config_accepts_minimal_supported_setup():
    config = _make_base_stream_config()
    assert config.moe_token_dispatcher_type == "stream"
    assert config.recompute_granularity == "selective"
    assert "moe" in config.recompute_modules
