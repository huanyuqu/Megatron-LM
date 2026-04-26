import argparse
from typing import Any

import torch
try:
    import pytest
except ModuleNotFoundError:  # pragma: no cover - script mode on lightweight envs.
    pytest = None

from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_submodules
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training.initialize import _set_random_seed
from tests.unit_tests.test_utilities import Utils


def _make_config(
    dispatcher_type: str,
    grouped_gemm: bool,
    stream_overlap: bool,
    stream_version: str,
    hidden_size: int,
    num_attention_heads: int,
    num_moe_experts: int,
    moe_router_topk: int,
    moe_ffn_hidden_size: int,
    dtype: torch.dtype,
    moe_aux_loss_coeff: float,
) -> TransformerConfig:
    return TransformerConfig(
        num_layers=1,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_moe_experts=num_moe_experts,
        use_cpu_initialization=False,
        moe_token_dispatcher_type=dispatcher_type,
        moe_stream_version=stream_version,
        moe_stream_overlap=stream_overlap,
        moe_router_load_balancing_type="aux_loss",
        moe_router_topk=moe_router_topk,
        moe_aux_loss_coeff=moe_aux_loss_coeff,
        moe_grouped_gemm=grouped_gemm,
        moe_ffn_hidden_size=moe_ffn_hidden_size,
        add_bias_linear=False,
        tensor_model_parallel_size=1,
        expert_model_parallel_size=2,
        sequence_parallel=False,
        recompute_granularity="selective",
        recompute_modules=["moe"],
        moe_router_dtype="fp32",
        bf16=dtype == torch.bfloat16,
        fp16=dtype == torch.float16,
        params_dtype=dtype,
    )


def _build_moe_layer(config: TransformerConfig) -> MoELayer:
    submodules = get_gpt_layer_local_submodules(
        num_experts=config.num_moe_experts, moe_grouped_gemm=config.moe_grouped_gemm
    )
    moe_layer = MoELayer(config, submodules.mlp.submodules).cuda()
    moe_layer.train()
    moe_layer.set_layer_number(0)
    return moe_layer


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
    stream_version: str,
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
    grouped_gemm: bool = False,
    stream_overlap: bool = False,
    stream_version: str = "v1",
    seq_len: int = 16,
    micro_batch_size: int = 4,
    hidden_size: int = 64,
    moe_ffn_hidden_size: int = 128,
    num_moe_experts: int = 4,
    moe_router_topk: int = 2,
    dtype: torch.dtype = torch.float32,
    moe_aux_loss_coeff: float = 0.01,
) -> None:
    if torch.cuda.device_count() < 2:
        raise RuntimeError("Stream MoE EP smoke test requires at least 2 CUDA devices.")

    try:
        Utils.initialize_model_parallel(tensor_model_parallel_size=1, expert_model_parallel_size=2)
        _set_random_seed(seed_=123, data_parallel_random_init=False)
        num_attention_heads = max(1, hidden_size // 64)

        reference_config = _make_config(
            dispatcher_type="alltoall",
            grouped_gemm=grouped_gemm,
            stream_version="v1",
            stream_overlap=False,
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
    grouped_gemm: bool = False,
    stream_overlap: bool = False,
    stream_version: str = "v1",
    seq_len: int = 16,
    micro_batch_size: int = 4,
    hidden_size: int = 64,
    moe_ffn_hidden_size: int = 128,
    num_moe_experts: int = 4,
    moe_router_topk: int = 2,
    dtype: torch.dtype = torch.float32,
    moe_aux_loss_coeff: float = 0.01,
    warmup: int = 2,
    iters: int = 5,
) -> None:
    if torch.cuda.device_count() < 2:
        raise RuntimeError("Stream MoE benchmark requires at least 2 CUDA devices.")

    try:
        Utils.initialize_model_parallel(tensor_model_parallel_size=1, expert_model_parallel_size=2)
        _set_random_seed(seed_=123, data_parallel_random_init=False)
        num_attention_heads = max(1, hidden_size // 64)
        config = _make_config(
            dispatcher_type=dispatcher_type,
            grouped_gemm=grouped_gemm,
            stream_version=stream_version if dispatcher_type == "stream" else "v1",
            stream_overlap=stream_overlap and dispatcher_type == "stream",
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_moe_experts=num_moe_experts,
            moe_router_topk=moe_router_topk,
            moe_ffn_hidden_size=moe_ffn_hidden_size,
            dtype=dtype,
            moe_aux_loss_coeff=moe_aux_loss_coeff,
        )
        layer = _build_moe_layer(config)

        hidden_states_cpu = torch.randn(
            seq_len,
            micro_batch_size,
            hidden_size,
            dtype=dtype,
        )
        grad_output_cpu = torch.randn_like(hidden_states_cpu)

        def run_once():
            layer.zero_grad(set_to_none=True)
            hidden_states = hidden_states_cpu.cuda().detach().clone().requires_grad_(True)
            grad_output = grad_output_cpu.cuda()
            output, _ = layer(hidden_states)
            output.backward(grad_output)
            return output

        for _ in range(warmup):
            run_once()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        baseline = torch.cuda.memory_allocated()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            run_once()
        end.record()
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated()
        elapsed_ms = start.elapsed_time(end) / max(1, iters)

        if torch.distributed.get_rank() == 0:
            print(
                f"[stream benchmark] dispatcher={dispatcher_type} grouped_gemm={grouped_gemm} "
                f"stream_version={stream_version if dispatcher_type == 'stream' else 'v1'} "
                f"stream_overlap={stream_overlap and dispatcher_type == 'stream'} "
                f"dtype={dtype} shape=({seq_len},{micro_batch_size},{hidden_size}) "
                f"topk={moe_router_topk} num_experts={num_moe_experts} "
                f"ffn={moe_ffn_hidden_size} aux={moe_aux_loss_coeff} "
                f"baseline={baseline / 1024**2:.2f}MiB "
                f"peak={peak / 1024**2:.2f}MiB delta={(peak - baseline) / 1024**2:.2f}MiB "
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
    parser.add_argument("--grouped-gemm", action="store_true")
    parser.add_argument("--stream-version", choices=["v1", "v2"], default="v1")
    parser.add_argument("--stream-overlap", action="store_true")
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--micro-batch-size", type=int, default=4)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--ffn-hidden-size", type=int, default=128)
    parser.add_argument("--num-experts", type=int, default=4)
    parser.add_argument("--topk", type=int, default=2)
    parser.add_argument("--aux-loss-coeff", type=float, default=0.01)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=5)
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
        grouped_gemm=args.grouped_gemm,
        stream_version=args.stream_version,
        stream_overlap=args.stream_overlap,
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
            warmup=args.warmup,
            iters=args.iters,
            **common_kwargs,
        )


if __name__ == "__main__":
    main()
