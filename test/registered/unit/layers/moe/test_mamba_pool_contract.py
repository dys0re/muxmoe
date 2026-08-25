"""Constructor-contract guards for the Mamba pool wiring.

`HybridReqToTokenPool._init_mamba_pool` forwards a fixed kwarg set into
`mamba_pool_cls(...)`. Adding a kwarg on the caller side without adding the
parameter to `MambaPool.__init__` is a hard TypeError on *every* hybrid model
startup, and it is invisible to any test that does not construct a real pool
(which needs CUDA). These tests check the contract statically instead.
"""

import ast
import inspect
import textwrap

import torch

from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool, MambaPool


def _forwarded_mamba_pool_kwargs() -> set[str]:
    """Kwarg names `_init_mamba_pool` passes into `self.mamba_pool_cls(...)`."""
    source = textwrap.dedent(inspect.getsource(HybridReqToTokenPool._init_mamba_pool))
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        # Match `self.mamba_pool_cls(...)`.
        if (
            isinstance(func, ast.Attribute)
            and func.attr == "mamba_pool_cls"
            and isinstance(func.value, ast.Name)
            and func.value.id == "self"
        ):
            return {kw.arg for kw in node.keywords if kw.arg is not None}
    raise AssertionError("no self.mamba_pool_cls(...) call found in _init_mamba_pool")


def test_mamba_pool_accepts_every_forwarded_kwarg():
    forwarded = _forwarded_mamba_pool_kwargs()
    accepted = set(inspect.signature(MambaPool.__init__).parameters) - {"self"}
    missing = forwarded - accepted
    assert not missing, (
        f"_init_mamba_pool forwards kwargs MambaPool.__init__ does not accept: "
        f"{sorted(missing)}. This is a TypeError on every hybrid model startup."
    )


def test_slab_size_is_plumbed_end_to_end():
    # The opt-in slab size must survive the whole caller chain, otherwise the
    # server arg silently does nothing.
    for fn in (HybridReqToTokenPool.__init__, HybridReqToTokenPool._init_mamba_pool):
        assert "mamba_state_slab_size" in inspect.signature(fn).parameters
    assert "mamba_state_slab_size" in _forwarded_mamba_pool_kwargs()
    assert "mamba_state_slab_size" in inspect.signature(MambaPool.__init__).parameters


def test_slab_size_defaults_to_disabled():
    # Default must stay None so the contiguous pool path is byte-identical.
    for fn in (
        HybridReqToTokenPool.__init__,
        HybridReqToTokenPool._init_mamba_pool,
        MambaPool.__init__,
    ):
        assert inspect.signature(fn).parameters["mamba_state_slab_size"].default is None


def test_speculative_state_accessor_returns_the_allocated_state():
    pool = object.__new__(MambaPool)
    state = MambaPool.SpeculativeState(
        conv=[],
        temporal=torch.empty(0),
        intermediate_ssm=torch.empty(0),
        intermediate_conv_window=[],
    )
    pool.mamba_cache = state

    assert pool.get_speculative_mamba2_params_all_layers() is state
