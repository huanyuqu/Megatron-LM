import argparse
from contextlib import nullcontext
import threading
import time
from typing import Any

import torch
try:
    import pytest
except ModuleNotFoundError:  # pragma: no cover - script mode on lightweight envs.
    pytest = None

from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_local_spec,
    get_gpt_layer_local_submodules,
)
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training.initialize import _set_random_seed
from tests.unit_tests.test_utilities import Utils


def _make_config(
    dispatcher_type: str,
    grouped_gemm: bool,
    stream_overlap: bool,
    stream_v2_sparse_comm: bool,
    stream_v2_local_first: bool,
    stream_version: str,
    recompute_mode: str,
    recompute_num_layers: int | None,
    ep_size: int,
    num_layers: int,
    hidden_size: int,
    num_attention_heads: int,
    num_moe_experts: int,
    moe_router_topk: int,
    moe_ffn_hidden_size: int,
    dtype: torch.dtype,
    moe_aux_loss_coeff: float,
) -> TransformerConfig:
    if recompute_mode == "moe":
        recompute_kwargs = dict(
            recompute_granularity="selective",
            recompute_method=None,
            recompute_num_layers=None,
            recompute_modules=["moe"],
        )
    elif recompute_mode == "attn_moe":
        recompute_kwargs = dict(
            recompute_granularity="selective",
            recompute_method=None,
            recompute_num_layers=None,
            recompute_modules=["core_attn", "moe"],
        )
    elif recompute_mode == "full":
        recompute_kwargs = dict(
            recompute_granularity="full",
            recompute_method="uniform",
            recompute_num_layers=recompute_num_layers or 1,
            recompute_modules=None,
        )
    elif recompute_mode == "none":
        recompute_kwargs = dict(
            recompute_granularity=None,
            recompute_method=None,
            recompute_num_layers=None,
            recompute_modules=None,
        )
    else:
        raise ValueError(f"Unsupported recompute_mode: {recompute_mode}")

    return TransformerConfig(
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_moe_experts=num_moe_experts,
        use_cpu_initialization=False,
        moe_token_dispatcher_type=dispatcher_type,
        moe_stream_version=stream_version,
        moe_stream_overlap=stream_overlap,
        moe_stream_v2_sparse_comm=stream_v2_sparse_comm,
        moe_stream_v2_local_first=stream_v2_local_first,
        moe_router_load_balancing_type="aux_loss",
        moe_router_topk=moe_router_topk,
        moe_aux_loss_coeff=moe_aux_loss_coeff,
        moe_grouped_gemm=grouped_gemm,
        moe_ffn_hidden_size=moe_ffn_hidden_size,
        add_bias_linear=False,
        tensor_model_parallel_size=1,
        expert_model_parallel_size=ep_size,
        sequence_parallel=False,
        **recompute_kwargs,
        moe_router_dtype="fp32",
        bf16=dtype == torch.bfloat16,
        fp16=dtype == torch.float16,
        params_dtype=dtype,
    )


def _build_moe_layer(config: TransformerConfig, layer_number: int = 0) -> MoELayer:
    submodules = get_gpt_layer_local_submodules(
        num_experts=config.num_moe_experts, moe_grouped_gemm=config.moe_grouped_gemm
    )
    moe_layer = MoELayer(config, submodules.mlp.submodules).cuda()
    moe_layer.train()
    moe_layer.set_layer_number(layer_number)
    return moe_layer


def _build_moe_stack(config: TransformerConfig) -> torch.nn.ModuleList:
    return torch.nn.ModuleList(
        [_build_moe_layer(config, layer_number=i + 1) for i in range(config.num_layers)]
    ).cuda()


def _build_transformer_stack(config: TransformerConfig) -> TransformerBlock:
    block = TransformerBlock(
        config,
        get_gpt_layer_local_spec(
            num_experts=config.num_moe_experts,
            moe_grouped_gemm=config.moe_grouped_gemm,
        ),
    ).cuda()
    if config.params_dtype != torch.float32:
        block = block.to(dtype=config.params_dtype)
    return block


def _build_gpt_stack(config: TransformerConfig, max_sequence_length: int) -> GPTModel:
    model = GPTModel(
        config=config,
        transformer_layer_spec=get_gpt_layer_local_spec(
            num_experts=config.num_moe_experts,
            moe_grouped_gemm=config.moe_grouped_gemm,
        ),
        vocab_size=32000,
        max_sequence_length=max_sequence_length,
        pre_process=False,
        post_process=False,
        position_embedding_type="none",
    ).cuda()
    if config.params_dtype != torch.float32:
        model = model.to(dtype=config.params_dtype)
    model.train()
    return model


def _activation_offload_context(enabled: bool):
    if not enabled:
        return nullcontext()

    def pack(tensor: torch.Tensor):
        if tensor.is_cuda:
            return tensor.device, tensor.detach().cpu()
        return None, tensor

    def unpack(packed):
        device, tensor = packed
        if device is None:
            return tensor
        return tensor.to(device, non_blocking=True)

    return torch.autograd.graph.saved_tensors_hooks(pack, unpack)


def _is_rank_zero() -> bool:
    return not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0


def _print_close_stats(name: str, reference: torch.Tensor, candidate: torch.Tensor) -> None:
    diff = (reference.float() - candidate.float()).abs()
    ref_abs = reference.float().abs()
    max_abs = diff.max()
    max_ref = ref_abs.max()
    rel_l2 = diff.norm() / ref_abs.norm().clamp_min(1e-12)
    denom = torch.maximum(ref_abs, torch.ones_like(ref_abs) * 1e-12)
    max_rel = (diff / denom).max()
    mean_abs = diff.mean()
    if _is_rank_zero():
        print(
            f"[stream compare] {name}: max_abs={max_abs.item():.8f} "
            f"mean_abs={mean_abs.item():.8f} rel_l2={rel_l2.item():.8f} "
            f"max_rel={max_rel.item():.8f} "
            f"max_ref={max_ref.item():.8f}",
            flush=True,
        )


def _assert_parameter_grad_close(
    name: str,
    reference: torch.Tensor,
    candidate: torch.Tensor,
    *,
    dtype: torch.dtype,
    atol: float,
    rtol: float,
) -> None:
    ref = reference.float()
    cand = candidate.float()
    if dtype == torch.float32:
        torch.testing.assert_close(
            ref,
            cand,
            atol=atol,
            rtol=rtol,
            msg=f"Gradient mismatch for parameter {name}",
        )
        return

    # In bf16/fp16 the two dispatchers can accumulate the same mathematical
    # gradient in a different order. Elementwise max relative error is too
    # brittle for router/expert gradient tensors, especially near zeros, so use
    # a norm-level criterion while keeping forward/input-grad elementwise checks.
    rel_l2 = (ref - cand).norm() / ref.norm().clamp_min(1e-12)
    norm_rtol = 2e-2
    if rel_l2 > norm_rtol:
        raise AssertionError(
            f"Gradient mismatch for parameter {name}: "
            f"relative L2 error {rel_l2.item():.6f} exceeds {norm_rtol:.6f}"
        )


def _clone_state_dict_to_cpu(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    cloned: dict[str, Any] = {}
    for name, tensor in module.state_dict().items():
        if torch.is_tensor(tensor):
            cloned[name] = tensor.detach().cpu().clone()
        else:
            cloned[name] = tensor
    return cloned


def _measure_dispatcher(
    dispatcher_type: str,
    state_dict: dict[str, torch.Tensor],
    hidden_states_cpu: torch.Tensor,
    grad_output_cpu: torch.Tensor,
    *,
    grouped_gemm: bool,
    stream_overlap: bool,
    stream_v2_sparse_comm: bool,
    stream_v2_local_first: bool,
    stream_version: str,
    recompute_mode: str,
    recompute_num_layers: int | None,
    ep_size: int,
    num_layers: int,
    hidden_size: int,
    num_attention_heads: int,
    num_moe_experts: int,
    moe_router_topk: int,
    moe_ffn_hidden_size: int,
    dtype: torch.dtype,
    moe_aux_loss_coeff: float,
) -> dict[str, Any]:
    config = _make_config(
        dispatcher_type=dispatcher_type,
        grouped_gemm=grouped_gemm,
        stream_version=stream_version,
        stream_overlap=stream_overlap,
        stream_v2_sparse_comm=stream_v2_sparse_comm,
        stream_v2_local_first=stream_v2_local_first,
        recompute_mode=recompute_mode,
        recompute_num_layers=recompute_num_layers,
        ep_size=ep_size,
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_moe_experts=num_moe_experts,
        moe_router_topk=moe_router_topk,
        moe_ffn_hidden_size=moe_ffn_hidden_size,
        dtype=dtype,
        moe_aux_loss_coeff=moe_aux_loss_coeff,
    )
    layer = _build_moe_layer(config)
    layer.load_state_dict(state_dict)

    hidden_states = hidden_states_cpu.cuda().detach().clone().requires_grad_(True)
    grad_output = grad_output_cpu.cuda()

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_allocated()
    output, _ = layer(hidden_states)
    output.backward(grad_output)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()

    result = {
        "output": output.detach().cpu(),
        "input_grad": hidden_states.grad.detach().cpu(),
        "param_grads": {
            name: param.grad.detach().cpu().clone() for name, param in layer.named_parameters()
        },
        "baseline": baseline,
        "peak": peak,
        "delta": peak - baseline,
    }

    del output
    del hidden_states
    del layer
    torch.cuda.empty_cache()
    return result


def run_stream_vs_alltoall_parity(
    *,
    ep_size: int = 2,
    num_layers: int = 1,
    grouped_gemm: bool = False,
    stream_overlap: bool = False,
    stream_v2_sparse_comm: bool = False,
    stream_v2_local_first: bool = True,
    stream_version: str = "v1",
    recompute_mode: str = "moe",
    recompute_num_layers: int | None = None,
    seq_len: int = 16,
    micro_batch_size: int = 4,
    hidden_size: int = 64,
    moe_ffn_hidden_size: int = 128,
    num_moe_experts: int = 4,
    moe_router_topk: int = 2,
    dtype: torch.dtype = torch.float32,
    moe_aux_loss_coeff: float = 0.01,
) -> None:
    if torch.cuda.device_count() < ep_size:
        raise RuntimeError(f"Stream MoE EP smoke test requires at least {ep_size} CUDA devices.")

    try:
        Utils.initialize_model_parallel(tensor_model_parallel_size=1, expert_model_parallel_size=ep_size)
        _set_random_seed(seed_=123, data_parallel_random_init=False)
        num_attention_heads = max(1, hidden_size // 64)

        reference_config = _make_config(
            dispatcher_type="alltoall",
            grouped_gemm=grouped_gemm,
            stream_version="v1",
            stream_overlap=False,
            stream_v2_sparse_comm=False,
            stream_v2_local_first=True,
            recompute_mode="moe",
            recompute_num_layers=None,
            ep_size=ep_size,
            num_layers=1,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_moe_experts=num_moe_experts,
            moe_router_topk=moe_router_topk,
            moe_ffn_hidden_size=moe_ffn_hidden_size,
            dtype=dtype,
            moe_aux_loss_coeff=moe_aux_loss_coeff,
        )
        reference_layer = _build_moe_layer(reference_config)
        state_dict = _clone_state_dict_to_cpu(reference_layer)
        del reference_layer
        torch.cuda.empty_cache()

        hidden_states_cpu = torch.randn(
            seq_len,
            micro_batch_size,
            hidden_size,
            dtype=dtype,
        )
        grad_output_cpu = torch.randn_like(hidden_states_cpu)

        alltoall_result = _measure_dispatcher(
            "alltoall",
            state_dict,
            hidden_states_cpu,
            grad_output_cpu,
            grouped_gemm=grouped_gemm,
            stream_version="v1",
            stream_overlap=False,
            stream_v2_sparse_comm=False,
            stream_v2_local_first=True,
            recompute_mode=recompute_mode,
            recompute_num_layers=recompute_num_layers,
            ep_size=ep_size,
            num_layers=num_layers,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_moe_experts=num_moe_experts,
            moe_router_topk=moe_router_topk,
            moe_ffn_hidden_size=moe_ffn_hidden_size,
            dtype=dtype,
            moe_aux_loss_coeff=moe_aux_loss_coeff,
        )
        stream_result = _measure_dispatcher(
            "stream",
            state_dict,
            hidden_states_cpu,
            grad_output_cpu,
            grouped_gemm=grouped_gemm,
            stream_version=stream_version,
            stream_overlap=stream_overlap,
            stream_v2_sparse_comm=stream_v2_sparse_comm,
            stream_v2_local_first=stream_v2_local_first,
            recompute_mode=recompute_mode,
            recompute_num_layers=recompute_num_layers,
            ep_size=ep_size,
            num_layers=num_layers,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_moe_experts=num_moe_experts,
            moe_router_topk=moe_router_topk,
            moe_ffn_hidden_size=moe_ffn_hidden_size,
            dtype=dtype,
            moe_aux_loss_coeff=moe_aux_loss_coeff,
        )

        atol = 1e-4 if dtype == torch.float32 else 2e-2
        rtol = 1e-4 if dtype == torch.float32 else 2e-2
        _print_close_stats("output", alltoall_result["output"], stream_result["output"])
        torch.testing.assert_close(
            alltoall_result["output"].float(),
            stream_result["output"].float(),
            atol=atol,
            rtol=rtol,
            msg="Forward outputs differ between alltoall and stream dispatchers",
        )
        _print_close_stats("input_grad", alltoall_result["input_grad"], stream_result["input_grad"])
        torch.testing.assert_close(
            alltoall_result["input_grad"].float(),
            stream_result["input_grad"].float(),
            atol=atol,
            rtol=rtol,
            msg="Input gradients differ between alltoall and stream dispatchers",
        )
        for name, ref_grad in alltoall_result["param_grads"].items():
            cand_grad = stream_result["param_grads"][name]
            _print_close_stats(f"param_grad:{name}", ref_grad, cand_grad)
            _assert_parameter_grad_close(
                name,
                ref_grad,
                cand_grad,
                dtype=dtype,
                atol=atol,
                rtol=rtol,
            )

        if torch.distributed.get_rank() == 0:
            print(
                f"[stream smoke] grouped_gemm={grouped_gemm} dtype={dtype} "
                f"stream_version={stream_version} "
                f"stream_overlap={stream_overlap} "
                f"stream_v2_sparse_comm={stream_v2_sparse_comm} "
                f"shape=({seq_len},{micro_batch_size},{hidden_size}) topk={moe_router_topk} "
                f"num_experts={num_moe_experts} ffn={moe_ffn_hidden_size} "
                f"aux={moe_aux_loss_coeff} "
                f"alltoall_baseline={alltoall_result['baseline'] / 1024**2:.2f}MiB "
                f"alltoall_peak={alltoall_result['peak'] / 1024**2:.2f}MiB "
                f"alltoall_delta={alltoall_result['delta'] / 1024**2:.2f}MiB "
                f"stream_baseline={stream_result['baseline'] / 1024**2:.2f}MiB "
                f"stream_peak={stream_result['peak'] / 1024**2:.2f}MiB "
                f"stream_delta={stream_result['delta'] / 1024**2:.2f}MiB"
            )
    finally:
        Utils.destroy_model_parallel()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


def run_single_dispatcher_benchmark(
    *,
    dispatcher_type: str,
    layer_kind: str = "moe",
    ep_size: int = 2,
    num_layers: int = 1,
    grouped_gemm: bool = False,
    stream_overlap: bool = False,
    stream_v2_sparse_comm: bool = False,
    stream_v2_local_first: bool = True,
    stream_version: str = "v1",
    recompute_mode: str = "moe",
    recompute_num_layers: int | None = None,
    seq_len: int = 16,
    micro_batch_size: int = 4,
    hidden_size: int = 64,
    moe_ffn_hidden_size: int = 128,
    num_moe_experts: int = 4,
    moe_router_topk: int = 2,
    dtype: torch.dtype = torch.float32,
    moe_aux_loss_coeff: float = 0.01,
    activation_offload: bool = False,
    grad_accum_steps: int = 1,
    warmup: int = 2,
    iters: int = 5,
    sample_device_memory_peak: bool = False,
) -> None:
    if torch.cuda.device_count() < ep_size:
        raise RuntimeError(f"Stream MoE benchmark requires at least {ep_size} CUDA devices.")

    try:
        Utils.initialize_model_parallel(tensor_model_parallel_size=1, expert_model_parallel_size=ep_size)
        _set_random_seed(seed_=123, data_parallel_random_init=False)
        num_attention_heads = max(1, hidden_size // 64)
        config = _make_config(
            dispatcher_type=dispatcher_type,
            grouped_gemm=grouped_gemm,
            stream_version=stream_version if dispatcher_type == "stream" else "v1",
            stream_overlap=stream_overlap and dispatcher_type == "stream",
            stream_v2_sparse_comm=stream_v2_sparse_comm and dispatcher_type == "stream",
            stream_v2_local_first=stream_v2_local_first,
            recompute_mode=recompute_mode,
            recompute_num_layers=recompute_num_layers,
            ep_size=ep_size,
            num_layers=num_layers,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_moe_experts=num_moe_experts,
            moe_router_topk=moe_router_topk,
            moe_ffn_hidden_size=moe_ffn_hidden_size,
            dtype=dtype,
            moe_aux_loss_coeff=moe_aux_loss_coeff,
        )
        if layer_kind == "moe":
            model = _build_moe_stack(config)
        elif layer_kind == "transformer":
            model = _build_transformer_stack(config)
        elif layer_kind == "gpt":
            model = _build_gpt_stack(config, max_sequence_length=seq_len)
        else:
            raise ValueError(f"Unsupported layer_kind: {layer_kind}")

        hidden_states_cpu = torch.randn(
            seq_len,
            micro_batch_size,
            hidden_size,
            dtype=dtype,
        )
        grad_output_cpu = torch.randn_like(hidden_states_cpu)
        attention_mask = torch.ones(
            (1, 1, seq_len, seq_len),
            dtype=torch.bool,
            device="cuda",
        )
        input_ids = torch.empty(
            (micro_batch_size, seq_len),
            dtype=torch.long,
            device="cuda",
        )
        position_ids = (
            torch.arange(seq_len, dtype=torch.long, device="cuda")
            .unsqueeze(0)
            .expand(micro_batch_size, -1)
        )

        grad_accum_steps = max(1, int(grad_accum_steps))

        def run_once():
            model.zero_grad(set_to_none=True)
            output = None
            with _activation_offload_context(activation_offload):
                for _ in range(grad_accum_steps):
                    hidden_states = hidden_states_cpu.cuda().detach().clone().requires_grad_(True)
                    grad_output = grad_output_cpu.cuda()
                    if layer_kind == "moe":
                        output = hidden_states
                        for layer in model:
                            output, _ = layer(output)
                    elif layer_kind == "gpt":
                        model.set_input_tensor(hidden_states)
                        output = model(
                            input_ids=input_ids,
                            position_ids=position_ids,
                            attention_mask=attention_mask,
                        )
                    else:
                        output = model(hidden_states=hidden_states, attention_mask=attention_mask)
                    output.backward(grad_output)
            return output

        for _ in range(warmup):
            run_once()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        baseline = torch.cuda.memory_allocated()
        baseline_reserved = torch.cuda.memory_reserved()
        try:
            free_memory, total_memory = torch.cuda.mem_get_info()
            baseline_device = total_memory - free_memory
        except RuntimeError:
            baseline_device = baseline_reserved

        torch.cuda.synchronize()
        device_index = torch.cuda.current_device()
        device_used = baseline_device
        stop_sampling = threading.Event() if sample_device_memory_peak else None

        def sample_device_peak() -> None:
            nonlocal device_used
            with torch.cuda.device(device_index):
                assert stop_sampling is not None
                while not stop_sampling.is_set():
                    try:
                        free_memory, total_memory = torch.cuda.mem_get_info()
                        device_used = max(device_used, total_memory - free_memory)
                    except RuntimeError:
                        pass
                    time.sleep(0.005)

        sampler = None
        if sample_device_memory_peak:
            sampler = threading.Thread(target=sample_device_peak, daemon=True)
            sampler.start()
        start_time = time.perf_counter()
        try:
            for _ in range(iters):
                run_once()
            torch.cuda.synchronize()
        finally:
            if stop_sampling is not None:
                stop_sampling.set()
            if sampler is not None:
                sampler.join()
        peak = torch.cuda.max_memory_allocated()
        peak_reserved = torch.cuda.max_memory_reserved()
        try:
            free_memory, total_memory = torch.cuda.mem_get_info()
            device_used = max(device_used, total_memory - free_memory)
        except RuntimeError:
            device_used = max(device_used, peak_reserved)
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0 / max(1, iters)
        delta = peak - baseline
        delta_reserved = peak_reserved - baseline_reserved
        delta_device = device_used - baseline_device

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            stats = torch.tensor(
                [
                    float(baseline),
                    float(peak),
                    float(delta),
                    float(baseline_reserved),
                    float(peak_reserved),
                    float(delta_reserved),
                    float(baseline_device),
                    float(device_used),
                    float(delta_device),
                    float(elapsed_ms),
                ],
                dtype=torch.float64,
                device=f"cuda:{torch.cuda.current_device()}",
            )
            torch.distributed.all_reduce(stats, op=torch.distributed.ReduceOp.MAX)
            (
                baseline,
                peak,
                delta,
                baseline_reserved,
                peak_reserved,
                delta_reserved,
                baseline_device,
                device_used,
                delta_device,
                elapsed_ms,
            ) = stats.tolist()

        if torch.distributed.get_rank() == 0:
            print(
                f"[stream benchmark] dispatcher={dispatcher_type} ep_size={ep_size} "
                f"layer_kind={layer_kind} num_layers={num_layers} grouped_gemm={grouped_gemm} "
                f"stream_version={stream_version if dispatcher_type == 'stream' else 'v1'} "
                f"stream_overlap={stream_overlap and dispatcher_type == 'stream'} "
                f"stream_v2_sparse_comm={stream_v2_sparse_comm and dispatcher_type == 'stream'} "
                f"stream_v2_local_first={stream_v2_local_first and dispatcher_type == 'stream'} "
                f"activation_offload={activation_offload} "
                f"grad_accum_steps={grad_accum_steps} "
                f"recompute_mode={recompute_mode} "
                f"recompute_num_layers={recompute_num_layers if recompute_mode == 'full' else None} "
                f"dtype={dtype} shape=({seq_len},{micro_batch_size},{hidden_size}) "
                f"topk={moe_router_topk} num_experts={num_moe_experts} "
                f"ffn={moe_ffn_hidden_size} aux={moe_aux_loss_coeff} "
                f"baseline={baseline / 1024**2:.2f}MiB "
                f"peak={peak / 1024**2:.2f}MiB delta={delta / 1024**2:.2f}MiB "
                f"reserved_baseline={baseline_reserved / 1024**2:.2f}MiB "
                f"reserved_peak={peak_reserved / 1024**2:.2f}MiB "
                f"reserved_delta={delta_reserved / 1024**2:.2f}MiB "
                f"device_baseline={baseline_device / 1024**2:.2f}MiB "
                f"device_used={device_used / 1024**2:.2f}MiB "
                f"device_delta={delta_device / 1024**2:.2f}MiB "
                f"time={elapsed_ms:.2f}ms"
            )
    finally:
        Utils.destroy_model_parallel()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if pytest is not None:

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA is required for stream smoke tests."
    )
    @pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires 2 GPUs.")
    @pytest.mark.parametrize("grouped_gemm", [False, True])
    def test_stream_matches_alltoall(grouped_gemm: bool):
        run_stream_vs_alltoall_parity(grouped_gemm=grouped_gemm)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["compare", "benchmark"],
        default="compare",
        help="compare checks alltoall vs stream parity; benchmark runs one dispatcher per process.",
    )
    parser.add_argument("--dispatcher", choices=["alltoall", "stream"], default="stream")
    parser.add_argument(
        "--layer-kind",
        choices=["moe", "transformer", "gpt"],
        default="moe",
        help=(
            "Benchmark only the MoE sublayer, direct TransformerBlock, or official "
            "GPTModel without embedding/output layers."
        ),
    )
    parser.add_argument("--ep-size", type=int, default=2)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--grouped-gemm", action="store_true")
    parser.add_argument("--stream-version", choices=["v1", "v2"], default="v1")
    parser.add_argument("--stream-overlap", action="store_true")
    parser.add_argument("--stream-v2-sparse-comm", action="store_true")
    parser.add_argument(
        "--stream-v2-no-local-first",
        action="store_true",
        help="Disable StreamMoE v2 current-rank-first path manipulation.",
    )
    parser.add_argument(
        "--activation-offload",
        action="store_true",
        help="Use saved-tensor CPU offload as an EP memory-saving baseline.",
    )
    parser.add_argument(
        "--grad-accum-steps",
        type=int,
        default=1,
        help="Run this many sequential microbatches per measured optimizer step.",
    )
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--micro-batch-size", type=int, default=4)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--ffn-hidden-size", type=int, default=128)
    parser.add_argument("--num-experts", type=int, default=4)
    parser.add_argument("--topk", type=int, default=2)
    parser.add_argument("--aux-loss-coeff", type=float, default=0.01)
    parser.add_argument(
        "--recompute-mode",
        choices=["moe", "attn_moe", "full", "none"],
        default="moe",
        help=(
            "Activation recompute policy. The default is selective MoE recompute; "
            "attn_moe additionally checkpoints core attention for model-level comparisons."
        ),
    )
    parser.add_argument(
        "--recompute-num-layers",
        type=int,
        default=None,
        help="Number of Transformer layers per full-checkpoint segment.",
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument(
        "--sample-device-memory-peak",
        action="store_true",
        help="Poll device memory during the measured loop to approximate nvidia-smi peak used memory.",
    )
    parser.add_argument(
        "--dtype", choices=["fp32", "bf16", "fp16"], default="fp32"
    )
    args = parser.parse_args()
    dtype_map = {
        "fp32": torch.float32,
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
    }
    common_kwargs = dict(
        ep_size=args.ep_size,
        num_layers=args.num_layers,
        grouped_gemm=args.grouped_gemm,
        stream_version=args.stream_version,
        stream_overlap=args.stream_overlap,
        stream_v2_sparse_comm=args.stream_v2_sparse_comm,
        stream_v2_local_first=not args.stream_v2_no_local_first,
        recompute_mode=args.recompute_mode,
        recompute_num_layers=args.recompute_num_layers,
        seq_len=args.seq_len,
        micro_batch_size=args.micro_batch_size,
        hidden_size=args.hidden_size,
        moe_ffn_hidden_size=args.ffn_hidden_size,
        num_moe_experts=args.num_experts,
        moe_router_topk=args.topk,
        dtype=dtype_map[args.dtype],
        moe_aux_loss_coeff=args.aux_loss_coeff,
    )
    if args.mode == "compare":
        run_stream_vs_alltoall_parity(**common_kwargs)
    else:
        run_single_dispatcher_benchmark(
            dispatcher_type=args.dispatcher,
            layer_kind=args.layer_kind,
            activation_offload=args.activation_offload,
            grad_accum_steps=args.grad_accum_steps,
            warmup=args.warmup,
            iters=args.iters,
            sample_device_memory_peak=args.sample_device_memory_peak,
            **common_kwargs,
        )


if __name__ == "__main__":
    main()
