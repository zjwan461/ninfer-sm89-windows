"""Launcher 参数模型：命令行构造、校验与 .bat 预设等价性。"""

from __future__ import annotations

from pathlib import Path

import pytest

from tools.launcher import config


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"ninfer")
    return path


def _values(tmp_path: Path, **overrides: object) -> dict[str, object]:
    engine = _touch(tmp_path / "bin" / "ninfer-serve.exe")
    artifact = _touch(tmp_path / "models" / "m.ninfer")
    values = config.default_values(str(engine))
    values["artifact"] = str(artifact)
    values.update(overrides)
    return values


# ------------------------------------------------------------------ 构造 argv


def test_build_argv_emits_only_set_fields(tmp_path: Path) -> None:
    values = _values(tmp_path)
    argv = config.build_argv(values)
    assert argv[0].endswith("ninfer-serve.exe")
    assert argv[1].endswith("m.ninfer")
    # 默认只带 host/port/kv-capacity，空字段一律不出现。
    assert "--host" in argv and "--port" in argv and "--kv-capacity" in argv
    for absent in ("--model-id", "--vision", "--spec", "--draft-tokens", "--prefill-chunk"):
        assert absent not in argv


def test_build_argv_bool_and_enum_order(tmp_path: Path) -> None:
    values = _values(
        tmp_path,
        model_id="qwen3.8-27b",
        vision=True,
        spec="mtp",
        draft_tokens=3,
        lm_head_draft=True,
        ngram="chain",
        max_context=262144,
    )
    argv = config.build_argv(values)
    assert argv[argv.index("--model-id") + 1] == "qwen3.8-27b"
    assert "--vision" in argv
    assert argv[argv.index("--spec") + 1] == "mtp"
    assert argv[argv.index("--draft-tokens") + 1] == "3"
    assert argv[argv.index("--max-context") + 1] == "262144"
    # 布尔开关不带值：它们后面紧跟的下一个 token 不可能是自己的值。
    assert argv[argv.index("--lm-head-draft") + 1] == "--ngram"


def test_build_argv_floats_are_compact(tmp_path: Path) -> None:
    argv = config.build_argv(_values(tmp_path, temperature=0.7, top_p=0.95))
    assert argv[argv.index("--temperature") + 1] == "0.7"
    assert argv[argv.index("--top-p") + 1] == "0.95"


def test_extra_args_pass_through(tmp_path: Path) -> None:
    argv = config.build_argv(_values(tmp_path, extra_args='--no-cuda-graph "E:\\a b\\x.json"'))
    assert argv[-2] == "--no-cuda-graph"
    assert argv[-1] == "E:\\a b\\x.json"


def test_directory_artifact_expands(tmp_path: Path) -> None:
    container = tmp_path / "models" / "qwen.ninfer"
    inner = _touch(container / "qwen.ninfer")
    values = _values(tmp_path)
    values["artifact"] = str(container)
    argv = config.build_argv(values)
    assert Path(argv[1]) == inner


# ------------------------------------------------------------------ 预设等价


def test_preset_matches_qwen_4080s_script() -> None:
    presets = {preset["name"]: preset["values"] for preset in config.builtin_presets()}
    values = presets["Qwen3.8-27B int8 · 4080s"]
    # 与 start-qwen38-rtx4080s.bat 的固定参数逐项对照。
    expected = {
        "model_id": "qwen3.8-27b",
        "max_context": 262144,
        "kv_capacity": "auto",
        "kv_dtype": "rk4v4-e8",
        "max_concurrency": 2,
        "max_pending_requests": 10,
        "pending_timeout_ms": 600000,
        "prefill_chunk": 1024,
        "default_max_tokens": 32768,
        "spec": "mtp",
        "draft_tokens": 3,
        "lm_head_draft": True,
        "ngram": "chain",
        "preserve_thinking": True,
        "device_state_slots": 3,
        "host_state_slots": 4,
        "host_kv_mib": 4096,
        "host": "127.0.0.1",
        "port": 8080,
    }
    for key, want in expected.items():
        assert values[key] == want, key
    assert values["artifact"].endswith("qwen3_8_27b_a8.ninfer")
    assert values["engine_binary"].endswith("build-4080s\\apps\\ninfer-serve.exe") or values["engine_binary"].endswith("ninfer-serve.exe")


def test_preset_4090_differs_only_in_documented_fields() -> None:
    presets = {preset["name"]: preset["values"] for preset in config.builtin_presets()}
    s4080 = presets["Qwen3.8-27B int8 · 4080s"]
    s4090 = presets["Qwen3.8-27B int8 · 4090"]
    differing = {key for key in s4080 if s4080[key] != s4090[key]}
    # .bat 注释声明的差异：prefill-chunk、max-context/max-concurrency、二进制路径。
    assert differing <= {
        "artifact", "engine_binary", "max_context", "max_concurrency", "prefill_chunk"
    }


def test_bonsai_preset_enables_vision() -> None:
    presets = {preset["name"]: preset["values"] for preset in config.builtin_presets()}
    values = presets["Bonsai2-27B VL · 4080s"]
    assert values["vision"] is True
    assert values["model_id"] == "bonsai-27b"
    assert values["max_concurrency"] == 3
    assert values["draft_tokens"] == 2


# ------------------------------------------------------------------ 校验


def test_validate_requires_artifact_and_engine(tmp_path: Path) -> None:
    values = config.default_values()
    values["engine_binary"] = ""
    values["artifact"] = ""
    errors, _ = config.validate(values)
    assert any("引擎" in error for error in errors)
    assert any("模型" in error for error in errors)


def test_validate_rejects_concurrency_and_dflash2_draft(tmp_path: Path) -> None:
    errors, _ = config.validate(_values(tmp_path, max_concurrency=9))
    assert any("1–8" in error for error in errors)

    errors, _ = config.validate(_values(tmp_path, spec="dflash2", draft_tokens=16))
    assert any("1–15" in error for error in errors)


def test_validate_requires_draft_tokens_with_spec(tmp_path: Path) -> None:
    errors, _ = config.validate(_values(tmp_path, spec="mtp"))
    assert any("draft-tokens" in error for error in errors)


def test_validate_cross_field_rules(tmp_path: Path) -> None:
    errors, _ = config.validate(_values(tmp_path, auto_save_evicted=True))
    assert any("slot-save-path" in error for error in errors)

    errors, _ = config.validate(_values(tmp_path, no_thinking=True, default_thinking_budget=512))
    assert any("default-thinking-budget" in error for error in errors)


def test_validate_warnings_do_not_block(tmp_path: Path) -> None:
    errors, warnings = config.validate(
        _values(tmp_path, draft_tokens=3, ngram_n=3, vision_max_tokens=4096)
    )
    assert errors == []
    joined = " ".join(warnings)
    assert "--spec" in joined and "n-gram" in joined and "--vision" in joined


def test_validate_context_ceiling_warns(tmp_path: Path) -> None:
    errors, warnings = config.validate(_values(tmp_path, max_context=999999))
    assert errors == []
    assert any("超过" in warning for warning in warnings)


# ------------------------------------------------------------------ 收敛与拆分


def test_normalize_coerces_types(tmp_path: Path) -> None:
    values, errors = config.normalize(
        {
            "max_concurrency": "4",
            "kv_capacity": "",
            "svision": None,
            "vision": "true",
            "kv_dtype": "fp8",
        }
    )
    assert errors == []
    assert values["max_concurrency"] == 4
    assert values["kv_capacity"] == "auto"
    assert values["vision"] is True
    assert values["kv_dtype"] == "fp8"


def test_normalize_rejects_bad_enum_and_number() -> None:
    _, errors = config.normalize({"kv_dtype": "int9"})
    assert any("kv-dtype" in error for error in errors)

    _, errors = config.normalize({"max_context": "abc"})
    assert any("整数" in error for error in errors)


def test_display_command_quotes_spaces(tmp_path: Path) -> None:
    argv = [str(tmp_path / "a b" / "ninfer-serve.exe"), str(tmp_path / "x y.ninfer")]
    assert '"' in config.display_command(argv)


@pytest.mark.parametrize(
    "text, expected",
    [
        ("", []),
        ("--no-cuda-graph", ["--no-cuda-graph"]),
        ('--chat-template "E:\\my dir\\t.jinja"', ["--chat-template", "E:\\my dir\\t.jinja"]),
    ],
)
def test_split_extra_args(text: str, expected: list[str]) -> None:
    assert config.split_extra_args(text) == expected


# ------------------------------------------------------------ 引擎构建标识


def test_engine_identity_prefers_variant_version(tmp_path: Path) -> None:
    variant = tmp_path / "dist" / "ninfer-sm89-windows-x64-0.6.1" / "sm80-rtx4080s"
    variant.mkdir(parents=True)
    (variant / "VERSION").write_text("0.6.1-rtx4080s\n", encoding="utf-8")
    exe = variant / "ninfer-serve.exe"
    exe.write_bytes(b"x")
    assert config.engine_identity(str(exe)) == "0.6.1-rtx4080s"


def test_engine_identity_never_reports_upstream_tag(tmp_path: Path) -> None:
    """build 目录没有 VERSION 时退化为目录名，不能拿仓库根的 0.6.1-rtx3090。"""

    apps = tmp_path / "build-4080s" / "apps"
    apps.mkdir(parents=True)
    exe = apps / "ninfer-serve.exe"
    exe.write_bytes(b"x")
    identity = config.engine_identity(str(exe))
    assert identity == "build-4080s"
    assert "rtx3090" not in identity


def test_engine_identity_empty_without_engine() -> None:
    assert config.engine_identity(None) == ""
    assert config.engine_identity("") == ""


# --------------------------------------------------------- artifact 能力校验


def _info(components: tuple[str, ...], proposal: bool = True) -> config.ArtifactInfo:
    return config.ArtifactInfo(path="x.ninfer", ok=True, components=components, proposal=proposal)


def test_capability_errors_blocks_missing_spec_component() -> None:
    """Bonsai 只有 mtp：--spec dflash / dflash2 必须被拦住（引擎会 FATAL: missing component）。"""

    bonsai = _info(("mtp", "proposal", "text", "vision"))
    errors = config.capability_errors({"spec": "dflash", "draft_tokens": 2}, bonsai)
    assert len(errors) == 1
    assert "dflash" in errors[0]
    assert "mtp" in errors[0]

    assert config.capability_errors({"spec": "mtp", "draft_tokens": 2}, bonsai) == []


def test_capability_errors_allows_qwen_dflash2() -> None:
    qwen = _info(("dflash2", "mtp", "proposal", "text", "vision"))
    assert config.capability_errors({"spec": "dflash2", "draft_tokens": 7}, qwen) == []
    # DFlash（v1）面向 35B-A3B，Qwen3.8 产物里没有该组件。
    errors = config.capability_errors({"spec": "dflash", "draft_tokens": 7}, qwen)
    assert len(errors) == 1 and "dflash2" in errors[0]


def test_capability_errors_vision_and_proposal() -> None:
    text_only = _info(("mtp", "text"), proposal=False)
    errors = config.capability_errors({"vision": True, "lm_head_draft": True, "spec": "mtp"}, text_only)
    assert any("vision" in error for error in errors)
    assert any("proposal" in error for error in errors)


def test_capability_errors_accepts_proposal_section() -> None:
    with_proposal = _info(("mtp", "text"), proposal=True)
    assert config.capability_errors({"lm_head_draft": True, "spec": "mtp", "draft_tokens": 3}, with_proposal) == []


def test_inspect_artifact_reports_error_for_non_artifact(tmp_path: Path) -> None:
    bogus = tmp_path / "bogus.ninfer"
    bogus.write_bytes(b"not a v3 artifact")
    info = config.inspect_artifact(str(bogus))
    assert info.ok is False
    assert info.error


def test_validate_warns_when_artifact_unreadable(tmp_path: Path) -> None:
    values = _values(tmp_path)  # _touch 写入的是假产物，读取器会拒绝
    errors, warnings = config.validate(values)
    assert not any("组件" in error for error in errors)
    assert any("跳过能力校验" in warning for warning in warnings)


@pytest.mark.parametrize(
    "filename, required, forbidden",
    [
        ("bonsai2_27b_vl_mtp_q4q5.ninfer", ("mtp", "vision"), ("dflash", "dflash2")),
        ("qwen3_8_27b_a8.ninfer", ("mtp", "dflash2"), ("dflash",)),
    ],
)
def test_local_artifacts_component_surface(
    filename: str, required: tuple[str, ...], forbidden: tuple[str, ...]
) -> None:
    """对真实产物核对组件面；本机没有该产物时跳过。"""

    path = config.resolve_artifact(filename)
    if not Path(path).is_file():
        pytest.skip(f"{filename} not present")
    info = config.inspect_artifact(path)
    assert info.ok, info.error
    for component in required:
        assert component in info.components, (filename, info.components)
    for component in forbidden:
        assert component not in info.components, (filename, info.components)
    assert info.proposal is True
    assert info.max_context is not None