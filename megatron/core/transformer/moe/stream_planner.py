# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class StreamRoundAssignment:
    """One round of StreamMoE execution."""

    expert_indices: torch.LongTensor
    probs: torch.Tensor

    @property
    def num_tokens(self) -> int:
        return int(self.expert_indices.numel())


@dataclass(frozen=True)
class StreamRoundPlan:
    """Compact top-k routing plus a K-round execution plan."""

    topk_probs: torch.Tensor
    topk_indices: torch.LongTensor
    rounds: tuple[StreamRoundAssignment, ...]

    @property
    def num_rounds(self) -> int:
        return len(self.rounds)

    @property
    def num_tokens(self) -> int:
        return int(self.topk_indices.shape[0])

    @property
    def topk(self) -> int:
        return int(self.topk_indices.shape[1])


def build_stream_round_plan(
    topk_probs: torch.Tensor,
    topk_indices: torch.LongTensor,
) -> StreamRoundPlan:
    """Build the default K-round StreamMoE plan.

    The v1 default simply executes one compact top-k column per round. This already
    satisfies the StreamMoE invariants:
    - exactly K rounds,
    - every round covers all tokens,
    - every token contributes to exactly one expert per round,
    - after K rounds each token has covered all routed experts exactly once.
    """

    if topk_indices.ndim != 2:
        raise ValueError(
            f"topk_indices must have shape [num_tokens, topk], got {tuple(topk_indices.shape)}."
        )
    if topk_probs.shape != topk_indices.shape:
        raise ValueError(
            "topk_probs must have the same shape as topk_indices, "
            f"got {tuple(topk_probs.shape)} and {tuple(topk_indices.shape)}."
        )

    rounds = tuple(
        StreamRoundAssignment(
            expert_indices=topk_indices[:, round_id].reshape(-1),
            probs=topk_probs[:, round_id].reshape(-1),
        )
        for round_id in range(topk_indices.shape[1])
    )
    return StreamRoundPlan(
        topk_probs=topk_probs,
        topk_indices=topk_indices,
        rounds=rounds,
    )
