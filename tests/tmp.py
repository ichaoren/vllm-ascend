# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Timing comparison for the `_project_output` padding buffer.

`_project_output` pads the attention output to the per-rank token count before
O projection. The old form allocated with `new_zeros`, which writes the whole
buffer, even though the copy that follows overwrites every real row -- only the
padding tail needs zeroing. This measures both forms at the shape DSV4.1 PCP=4
prefill actually uses (2044 real rows into a 2048-row buffer).

Timing only; `test_pad_tail_only_matches_full_zero` covers equivalence.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not (hasattr(torch, "npu") and torch.npu.is_available()),
    reason="requires an NPU",
)

# DSV4.1 PCP=4 prefill: 2048 padded tokens per rank, 2044 real (4 pad rows),
# 64 heads x 512 kv_lora. Matches ZerosLike/TensorMove in the profile.
PER_RANK = 2048
NUM_REAL = 2044
NUM_HEADS = 64
HEAD_DIM = 512
DTYPE = torch.bfloat16

WARMUP = 10
ITERS = 100


def _pad_full_zero(output: torch.Tensor, per_rank: int) -> torch.Tensor:
    """Old form: zero the whole buffer, then copy the real rows over it."""
    padded = output.new_zeros((per_rank, output.shape[1], output.shape[2]))
    padded[: output.shape[0]] = output
    return padded


def _pad_tail_only(output: torch.Tensor, per_rank: int) -> torch.Tensor:
    """New form: zero only the padding tail."""
    padded = output.new_empty((per_rank, output.shape[1], output.shape[2]))
    padded[output.shape[0] :].zero_()
    padded[: output.shape[0]] = output
    return padded


def _time_us(fn, output: torch.Tensor) -> float:
    for _ in range(WARMUP):
        fn(output, PER_RANK)
    torch.npu.synchronize()
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    start.record()
    for _ in range(ITERS):
        fn(output, PER_RANK)
    end.record()
    torch.npu.synchronize()
    return start.elapsed_time(end) * 1000.0 / ITERS


def test_pad_tail_only_matches_full_zero():
    output = torch.randn(NUM_REAL, NUM_HEADS, HEAD_DIM, dtype=DTYPE, device="npu")
    torch.testing.assert_close(
        _pad_tail_only(output, PER_RANK),
        _pad_full_zero(output, PER_RANK),
        rtol=0,
        atol=0,
    )


def test_pad_tail_only_is_not_slower():
    output = torch.randn(NUM_REAL, NUM_HEADS, HEAD_DIM, dtype=DTYPE, device="npu")
    full_us = _time_us(_pad_full_zero, output)
    tail_us = _time_us(_pad_tail_only, output)
    print(
        f"\n_project_output pad [{NUM_REAL} -> {PER_RANK}, {NUM_HEADS}x{HEAD_DIM}, {DTYPE}]"
        f"\n  new_zeros + copy (old): {full_us:8.2f} us"
        f"\n  new_empty + tail zero : {tail_us:8.2f} us"
        f"\n  saved                 : {full_us - tail_us:8.2f} us/call"
        f"  ({(full_us - tail_us) / full_us * 100:5.1f}%)"
        f"\n  x40 layers            : {(full_us - tail_us) * 40 / 1000:8.3f} ms/chunk"
    )
    assert tail_us <= full_us
