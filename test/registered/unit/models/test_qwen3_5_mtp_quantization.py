from types import SimpleNamespace

from sglang.srt.models.qwen3_5_mtp import _quantization_excludes_mtp


def test_detects_gptq_dynamic_mtp_exclusion():
    quant_config = SimpleNamespace(
        dynamic={
            "-:.*mtp.*": {},
            "-:.*shared_expert.*": {},
        }
    )

    assert _quantization_excludes_mtp(quant_config)


def test_detects_moe_wna16_full_config_mtp_exclusion():
    quant_config = SimpleNamespace(
        full_config={"dynamic": {"-:.*mtp.*": {}}}
    )

    assert _quantization_excludes_mtp(quant_config)


def test_does_not_treat_positive_mtp_override_as_exclusion():
    quant_config = SimpleNamespace(dynamic={"+:.*mtp.*": {"bits": 8}})

    assert not _quantization_excludes_mtp(quant_config)
