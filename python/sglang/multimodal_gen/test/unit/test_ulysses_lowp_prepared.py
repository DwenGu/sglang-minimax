# SPDX-License-Identifier: Apache-2.0
"""Request-owned Lowp resources and the Sage2 consumer contract."""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from flashinfer.comm import UlyssesQKV

from sglang.multimodal_gen.runtime.layers.attention.backends import (
    ulysses_lowp_v2g as backend,
)


@pytest.fixture
def fake_communicators(monkeypatch):
    created = []

    class Communicator:
        def __init__(self, group, **kwargs):
            self.group, self.kwargs = group, kwargs
            self.closed = False
            self.prepare_calls = 0
            self.previous_outputs = []
            created.append(self)

        def __enter__(self):
            return self

        def __exit__(self, *_):
            self.closed = True

        def prepare_qkv(self, shape, *, used_sequence):
            self.prepare_calls += 1
            return SimpleNamespace(qkv_shape=shape, used_sequence=used_sequence)

        def scatter_qkv(self, q, k, v, *, workspace, out):
            self.previous_outputs.append(out)
            assert tuple(q.shape) == workspace.qkv_shape
            if out is not None:
                return out
            b, length, heads, d = q.shape
            total, local_heads = length * 2, heads // 2
            return UlyssesQKV(
                torch.empty(b, total, local_heads, d, dtype=torch.int8),
                torch.empty(b, total, local_heads, d, dtype=torch.int8),
                torch.empty(b, d, local_heads, total, dtype=torch.float8_e4m3fn),
                torch.ones(b, local_heads, 4),
                torch.ones(b, local_heads, 1),
                torch.ones(b, local_heads, d),
                "sage2_sm90",
                total,
                workspace.used_sequence,
                q.dtype,
            )

    monkeypatch.setattr(backend, "UlyssesCommunicator", Communicator)
    monkeypatch.setattr(
        backend, "get_sp_group", lambda: SimpleNamespace(ulysses_group="group")
    )
    monkeypatch.setattr(torch.cuda.nvtx, "range", lambda _: nullcontext())
    monkeypatch.setattr(
        backend,
        "_a2a_staging_buffer",
        lambda _, shape, dtype, device: torch.empty(shape, dtype=dtype, device=device),
    )
    return created


def _impl():
    return backend.UlyssesLowpV2GImpl(2, 128, False, 128**-0.5)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_layers_share_preparation_and_outputs_with_correct_sage_dtype(
    fake_communicators, monkeypatch, dtype
):
    layers = [_impl(), _impl()]
    for layer in layers:
        monkeypatch.setattr(layer, "_route", lambda *args: (2, 3))
    consumed = []

    def attention(q, k, v, output, qscale, kscale, vscale, *args):
        consumed.append((q.shape, output.dtype, vscale))
        output.fill_(7)

    monkeypatch.setattr(backend, "_sage_kernel", lambda layout: attention)
    q = torch.zeros(2, 2, 128, dtype=dtype)
    with backend.prepare_ulysses_qkv(
        layers, qkv_shape=(1, *q.shape), used_sequence=3, dtype=dtype, device=q.device
    ):
        assert layers[0]._prepared_qkv is layers[1]._prepared_qkv
        for layer in layers * 2:
            result = layer.forward_varlen_ulysses(
                q, q, q, cu_seqlens_host=(0, 3, 4), max_seqlen=3
            )
            assert result.dtype == dtype
            assert result.shape == (4, 1, 128)
            assert torch.all(result[:3] == 7)
            assert torch.count_nonzero(result[3:]) == 0
        comm = fake_communicators[0]
        assert comm.prepare_calls == 1
        assert comm.kwargs["backend"] == "nccl"
        assert comm.kwargs["max_bytes"] == 3 * q.nbytes
        assert "max_elems" not in comm.kwargs
        assert comm.previous_outputs[0] is None
        assert all(
            out is layers[0]._prepared_qkv.encoded for out in comm.previous_outputs[1:]
        )
        assert len(consumed) == 4
    assert comm.closed
    assert all(layer._prepared_qkv is None for layer in layers)


def test_request_failure_clears_bindings_and_closes_comm(fake_communicators):
    layers = [_impl(), _impl()]
    with pytest.raises(RuntimeError, match="injected attention failure"):
        with backend.prepare_ulysses_qkv(
            layers,
            qkv_shape=(1, 2, 2, 128),
            used_sequence=3,
            dtype=torch.bfloat16,
            device="cpu",
        ):
            raise RuntimeError("injected attention failure")
    assert fake_communicators[0].closed
    assert all(layer._prepared_qkv is None for layer in layers)


def test_forward_never_implicitly_prepares_and_rejects_changed_live_length(
    fake_communicators, monkeypatch
):
    impl = _impl()
    monkeypatch.setattr(impl, "_route", lambda *args: (2, 3))
    q = torch.zeros(2, 2, 128, dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="must be prepared"):
        impl.forward_varlen_ulysses(q, q, q, cu_seqlens_host=(0, 3, 4), max_seqlen=3)
    assert fake_communicators == []
    with backend.prepare_ulysses_qkv(
        [impl], qkv_shape=(1, *q.shape), used_sequence=4, dtype=q.dtype, device=q.device
    ):
        with pytest.raises(ValueError, match="used_sequence"):
            impl.forward_varlen_ulysses(
                q, q, q, cu_seqlens_host=(0, 3, 4), max_seqlen=3
            )
        assert not fake_communicators[0].previous_outputs


def test_model_binds_only_ulysses_block_attention(fake_communicators, monkeypatch):
    from sglang.multimodal_gen.runtime.models.dits import minimax_h3 as model_module
    from sglang.multimodal_gen.runtime.platforms.interface import AttentionBackendEnum

    modules = []
    for prefix, gate in (
        ("blocks.0.attn", None),
        ("blocks.1.attn", None),
        ("token_refiner.blocks.0.attn", None),
        ("blocks.2.attn", torch.nn.Identity()),
    ):
        module = model_module.MiniMaxH3Attention.__new__(
            model_module.MiniMaxH3Attention
        )
        torch.nn.Module.__init__(module)
        module.prefix, module.to_gate_compress = prefix, gate
        module._attention_impl = _impl()
        modules.append(module)
    model = SimpleNamespace(
        _resolve_attention_backend_once=Mock(),
        _resolved_attention_backend=AttentionBackendEnum.ULYSSES_LOWP_V2G,
        modules=lambda: iter(modules),
        num_attention_heads=8,
        arch=SimpleNamespace(attention_head_dim=128),
    )
    monkeypatch.setattr(model_module, "get_ulysses_ctx", lambda: (2, 0))
    monkeypatch.setattr(model_module, "get_ring_ctx", lambda: (1, 0))
    monkeypatch.setattr(model_module, "get_tp_world_size", lambda: 2)
    with model_module.MiniMaxH3DiTModel.prepare_ulysses_attention(
        model,
        sequence_length=4,
        packed_seq_params={"cu_seqlens_q_host": (0, 3, 4), "max_seqlen_q": 3},
        device="cpu",
    ):
        state = modules[0]._attention_impl._prepared_qkv
        assert state.workspace.qkv_shape == (1, 2, 4, 128)
        assert modules[1]._attention_impl._prepared_qkv is state
        assert modules[2]._attention_impl._prepared_qkv is None
        # Gate weights do not activate sparse attention when Lowp was selected.
        assert modules[3]._attention_impl._prepared_qkv is state
    assert all(module._attention_impl._prepared_qkv is None for module in modules)
