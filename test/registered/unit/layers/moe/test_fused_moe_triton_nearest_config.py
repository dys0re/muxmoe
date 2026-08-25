import json
from types import SimpleNamespace

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


from sglang.srt.layers.moe.moe_runner.triton_utils import fused_moe_triton_config


def test_uses_nearest_expert_count_config(monkeypatch, tmp_path):
    config_dir = tmp_path / "configs" / "triton_3_6_0"
    config_dir.mkdir(parents=True)
    tuned_config = {"1": {"BLOCK_SIZE_M": 16}}
    config_file = (
        config_dir
        / "E=60,N=256,device_name=Test_GPU,dtype=int4_w4a16.json"
    )
    config_file.write_text(json.dumps(tuned_config))

    monkeypatch.setenv("SGLANG_MOE_CONFIG_DIR", str(tmp_path))
    monkeypatch.setattr(fused_moe_triton_config.triton, "__version__", "3.6.0")
    monkeypatch.setattr(
        fused_moe_triton_config,
        "get_server_args",
        lambda: SimpleNamespace(
            enable_deterministic_inference=False, moe_expert_offload_gb=1
        ),
    )
    monkeypatch.setattr(
        fused_moe_triton_config, "get_device_name", lambda: "Test GPU"
    )
    fused_moe_triton_config.get_moe_configs.cache_clear()

    try:
        assert fused_moe_triton_config.get_moe_configs(
            57, 256, "int4_w4a16"
        ) == {1: {"BLOCK_SIZE_M": 16}}
    finally:
        fused_moe_triton_config.get_moe_configs.cache_clear()


def test_does_not_use_nearest_expert_count_without_offload(monkeypatch, tmp_path):
    config_dir = tmp_path / "configs" / "triton_3_6_0"
    config_dir.mkdir(parents=True)
    config_file = (
        config_dir
        / "E=60,N=256,device_name=Test_GPU,dtype=int4_w4a16.json"
    )
    config_file.write_text(json.dumps({"1": {"BLOCK_SIZE_M": 16}}))

    monkeypatch.setenv("SGLANG_MOE_CONFIG_DIR", str(tmp_path))
    monkeypatch.setattr(fused_moe_triton_config.triton, "__version__", "3.6.0")
    monkeypatch.setattr(
        fused_moe_triton_config,
        "get_server_args",
        lambda: SimpleNamespace(
            enable_deterministic_inference=False, moe_expert_offload_gb=0
        ),
    )
    monkeypatch.setattr(
        fused_moe_triton_config, "get_device_name", lambda: "Test GPU"
    )
    fused_moe_triton_config.get_moe_configs.cache_clear()

    try:
        assert (
            fused_moe_triton_config.get_moe_configs(
                57, 256, "int4_w4a16"
            )
            is None
        )
    finally:
        fused_moe_triton_config.get_moe_configs.cache_clear()
