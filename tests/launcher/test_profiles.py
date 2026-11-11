"""Launcher 档案存储：增删改查、内置预设导入与 .bat 渲染。"""

from __future__ import annotations

from pathlib import Path

from tools.launcher import config
from tools.launcher.launcher import render_bat


def _store(tmp_path: Path) -> config.Store:
    return config.Store(tmp_path / "home")


def test_store_roundtrip(tmp_path: Path) -> None:
    store = _store(tmp_path)
    values = config.default_values("C:\\x\\ninfer-serve.exe")
    values["artifact"] = "C:\\x\\m.ninfer"
    values["max_concurrency"] = 4

    saved = store.save_profile("我的配置", values)
    assert saved["name"] == "我的配置"

    listed = store.list_profiles()
    assert [item["name"] for item in listed] == ["我的配置"]
    assert listed[0]["values"]["max_concurrency"] == 4
    assert store.last_profile == "我的配置"

    # 新实例重新读盘，验证持久化。
    reopened = config.Store(tmp_path / "home")
    assert reopened.get_profile("我的配置")["values"]["max_concurrency"] == 4

    assert reopened.delete_profile("我的配置") is True
    assert reopened.get_profile("我的配置") is None


def test_import_presets_is_idempotent(tmp_path: Path) -> None:
    store = _store(tmp_path)
    added = store.import_presets()
    assert added
    names = {profile["name"] for profile in store.list_profiles()}
    assert "Qwen3.8-27B int8 · 4080s" in names
    assert all(profile["builtin"] for profile in store.list_profiles())
    # 首次导入后默认选中第一个预设（4080s / Qwen），而不是最后一个。
    assert store.last_profile == "Qwen3.8-27B int8 · 4080s"

    assert store.import_presets() == []
    assert len(store.list_profiles()) == len(added)


def test_import_presets_keeps_active_profile(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.import_presets()
    values = store.get_profile("Qwen3.8-27B int8 · 4080s")["values"]
    store.save_profile("my-own", values)
    assert store.last_profile == "my-own"

    store.import_presets(overwrite=True)
    assert store.last_profile == "my-own"


def test_settings_persist_and_default(tmp_path: Path) -> None:
    store = _store(tmp_path)
    settings = store.settings
    assert settings["ui_port"] == config.UI_PORT
    store.update_settings({"ui_port": 8123, "models_roots": ["D:\\models"]})
    assert config.Store(tmp_path / "home").settings["ui_port"] == 8123
    assert config.Store(tmp_path / "home").settings["models_roots"] == ["D:\\models"]


def test_render_bat_is_runnable_shape(tmp_path: Path) -> None:
    values = config.default_values("C:\\bin\\ninfer-serve.exe")
    values["artifact"] = "C:\\models\\m.ninfer"
    values["model_id"] = "qwen3.8-27b"
    values["max_context"] = 262144
    values["spec"] = "mtp"
    values["draft_tokens"] = 3
    values["lm_head_draft"] = True
    values["vision"] = True

    text = render_bat("测试档案", values)
    assert text.startswith("@echo off\r\n")
    assert 'set "NINFER_SERVER=C:\\bin\\ninfer-serve.exe"' in text
    assert '"%NINFER_SERVER%" "C:\\models\\m.ninfer"' in text
    assert "--model-id qwen3.8-27b" in text
    assert "--vision" in text
    assert "%*" in text
    assert text.rstrip().endswith("pause")


def test_render_bat_escapes_percent(tmp_path: Path) -> None:
    values = config.default_values("C:\\bin\\ninfer-serve.exe")
    values["artifact"] = "C:\\models\\100%\\m.ninfer"
    text = render_bat("pct", values)
    assert "100%%" in text


def test_spec_choices_carry_panel_guidance() -> None:
    """面板对每个 --spec 选项都要能显示名称、说明和建议草稿数。"""

    field = config.FIELD_BY_KEY["spec"]
    assert set(field.choice_info or {}) == set(config.SPECS)
    for choice in config.SPECS:
        info = (field.choice_info or {})[choice]
        assert info.get("label"), f"--spec {choice} has no display label"
        assert info.get("note"), f"--spec {choice} has no guidance note"
        assert 1 <= int(info["draft_tokens"]) <= 15
    # DSpine 必须声明全词表 head 的限制，否则面板会允许 --lm-head-draft。
    assert "lm-head-draft" in (field.choice_info or {})["dspine"]["note"]


def test_import_presets_restores_a_deleted_builtin(tmp_path: Path) -> None:
    """启动时补齐内置预设：删掉一个内置项后再导入，它必须回来（用户自建项不受影响）。"""

    store = _store(tmp_path)
    names = store.import_presets()
    assert names
    store.save_profile("my-own", config.default_values("C:\\x\\ninfer-serve.exe"))
    victim = names[0]
    assert store.delete_profile(victim) is True

    restored = store.import_presets()
    assert victim in restored
    assert store.get_profile(victim) is not None
    assert store.get_profile("my-own") is not None
