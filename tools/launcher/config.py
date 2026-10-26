"""NInfer Launcher（WebUI）的配置模型。

这里是 serve 参数面的唯一来源：前端表单由 ``FIELDS`` 生成，校验与子进程命令行
也由同一份定义构造，避免 UI 与命令行漂移。

分层：

- ``Field`` / ``FIELDS``：参数元数据（分组、控件类型、校验范围、依赖关系）；
- ``validate``：语义校验，返回 (errors, warnings)；
- ``build_argv``：把参数字典变成 ``ninfer-serve.exe`` 的完整 argv；
- ``Store``：profile / 设置的持久化；
- ``builtin_presets``：仓库四个 ``start-*.bat`` 的等价档案。
"""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]

UI_PORT = 8100

#: 上下文超过该值时给出警告（Qwen3.8/Bonsai 的 artifact 声明上限，serve 会拒绝更大值）。
CONTEXT_CEILING = 262144

#: 模型下载使用的 Hugging Face 端点与镜像。
DEFAULT_HF_ENDPOINT = "https://huggingface.co"
HF_MIRRORS: tuple[tuple[str, str], ...] = (
    ("https://huggingface.co", "huggingface.co（官方）"),
    ("https://hf-mirror.com", "hf-mirror.com（国内镜像）"),
)

#: 「下载」页的官方产物清单。体积取自 Hugging Face 仓库里的实际文件大小
#: （2026-10 核对），``preset`` 是开箱可用的内置档案（空表示要用同架构档案换 artifact）。
MODEL_CATALOG: tuple[dict[str, Any], ...] = (
    {
        "label": "Ternary Bonsai 2 27B（文本+视觉+MTP）",
        "repo": "jgamboa/Ternary-Bonsai-2-27B-NInfer-4090",
        "filename": "bonsai2_27b_vl_mtp_q4q5.ninfer",
        "size_gib": 6.4,
        "preset": "Bonsai2-27B VL · 4080s",
        "note": "Q4/Q5 三元权重 + vision + MTP；只支持 --spec mtp（可直接用内置预设）",
    },
    {
        "label": "Qwen3.8-27B int8 prefill（推荐）",
        "repo": "jgamboa/Qwen3.8-27B-NInfer-4090",
        "filename": "qwen3_8_27b_a8.ninfer",
        "size_gib": 19.03,
        "preset": "Qwen3.8-27B int8 · 4080s",
        "note": "Q4/Q5 权重 + int8 prefill，带 MTP 与 DFlash2（可直接用内置预设）",
    },
    {
        "label": "Qwen3.8-27B 官方产物（BF16 prefill）",
        "repo": "neroued/Qwen3.8-27B-NInfer",
        "filename": "qwen3_8_27b.ninfer",
        "size_gib": 19.03,
        "preset": "",
        "note": "与 a8 版同一份权重，prefill 走 BF16（更慢）；载入 Qwen3.8 预设后换 artifact 即可",
    },
    {
        "label": "Swift 1.5 Qwen3.8-27B（少思考微调）",
        "repo": "jgamboa/Swift-1.5-Qwen3.8-27B-NInfer-4090",
        "filename": "swift15_27b_a8.ninfer",
        "size_gib": 19.03,
        "preset": "",
        "note": "UkisAI 微调，形状与 Qwen3.8 相同；载入 Qwen3.8 预设后换 artifact（建议 --model-id swift-1.5）",
    },
    {
        "label": "Qwen3.6-35B-A3B（MoE，带 DFlash）",
        "repo": "neroued/Qwen3.6-35B-A3B-NInfer",
        "filename": "qwen3_6_35b_a3b.ninfer",
        "size_gib": 21.22,
        "preset": "",
        "note": "MoE 产物：投机要用 --spec dflash --draft-tokens 7（没有 MTP）；本机 32 GB 需压小上下文",
    },
)

GROUPS: tuple[tuple[str, str], ...] = (
    ("model", "模型"),
    ("capacity", "容量"),
    ("perf", "性能与投机解码"),
    ("cache", "KV 与会话缓存"),
    ("request", "请求默认值"),
    ("network", "网络、日志与媒体"),
)

KV_DTYPES = ("bf16", "int8", "fp8", "nvfp4", "k8v4", "rk8v4", "rk4v4", "rk4v4-e8", "rk2v4-e8")
SPECS = ("mtp", "dflash", "dflash2")
LOG_LEVELS = ("trace", "debug", "info", "warning", "error", "critical", "off")


@dataclass(frozen=True)
class Field:
    """一个 serve 参数的元数据。``cli=None`` 表示不是命令行开关（位置参数/原样追加）。"""

    key: str
    group: str
    label: str
    kind: str = "text"  # text | int | float | bool | enum | path | auto_int | raw | secret
    cli: str | None = None
    default: Any = None
    choices: tuple[str, ...] = ()
    minimum: float | None = None
    maximum: float | None = None
    help: str = ""
    advanced: bool = False
    allow_empty: bool = True
    placeholder: str = ""
    #: 条件显示：{"key": "<字段>", "values": [...]} 或 {"key": "<字段>", "nonempty": True}
    show_if: dict[str, Any] | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "group": self.group,
            "label": self.label,
            "kind": self.kind,
            "cli": self.cli,
            "default": self.default,
            "choices": list(self.choices),
            "min": self.minimum,
            "max": self.maximum,
            "help": self.help,
            "advanced": self.advanced,
            "allow_empty": self.allow_empty,
            "placeholder": self.placeholder,
            "show_if": self.show_if,
        }


FIELDS: tuple[Field, ...] = (
    # ---------------------------------------------------------------- 模型
    Field(
        "engine_binary", "model", "引擎（ninfer-serve.exe）", kind="path", default="",
        help="启动的子进程；本机默认 build-4080s\\apps\\ninfer-serve.exe",
    ),
    Field(
        "artifact", "model", "模型文件（.ninfer）", kind="path", default="",
        help="v3 .ninfer 产物；也可选包含它的目录",
    ),
    Field(
        "model_id", "model", "模型别名（--model-id）", cli="--model-id",
        placeholder="留空 = artifact 的 metadata.name",
    ),
    Field(
        "chat_template", "model", "聊天模板覆盖（--chat-template）", kind="path",
        cli="--chat-template", advanced=True,
    ),
    Field(
        "vision", "model", "启用视觉（--vision）", kind="bool", cli="--vision",
        help="启动期固定；artifact 需含 Vision 组件，之后无法再开启",
    ),
    Field(
        "vision_max_tokens", "model", "视觉 token 上限（--vision-max-tokens）", kind="int",
        cli="--vision-max-tokens", minimum=1, advanced=True, placeholder="serve 默认 8192",
        show_if={"key": "vision", "values": [True]},
    ),
    # ---------------------------------------------------------------- 容量
    Field(
        "max_context", "capacity", "最大上下文（--max-context）", kind="int", cli="--max-context",
        minimum=1, placeholder="如 262144（artifact 上限）",
    ),
    Field(
        "kv_capacity", "capacity", "KV 池容量（--kv-capacity）", kind="auto_int",
        cli="--kv-capacity", default="auto",
        help="auto = 按实测空闲显存测算，并保留 1024 MiB 余量",
    ),
    Field(
        "kv_dtype", "capacity", "KV 数据类型（--kv-dtype）", kind="enum", cli="--kv-dtype",
        choices=KV_DTYPES, placeholder="留空 = serve/artifact 默认",
    ),
    Field(
        "max_concurrency", "capacity", "最大并发（--max-concurrency）", kind="int",
        cli="--max-concurrency", default=2, minimum=1, maximum=8,
        help="产品上限 1–8；与 --max-context 共同决定 KV 池占用",
    ),
    Field(
        "max_pending_requests", "capacity", "排队上限（--max-pending-requests）", kind="int",
        cli="--max-pending-requests", minimum=0, placeholder="serve 默认",
    ),
    Field(
        "pending_timeout_ms", "capacity", "排队超时 ms（--pending-timeout-ms）", kind="int",
        cli="--pending-timeout-ms", minimum=0, placeholder="serve 默认",
    ),
    # ---------------------------------------------------------------- 性能
    Field(
        "prefill_chunk", "perf", "prefill 分块（--prefill-chunk）", kind="int",
        cli="--prefill-chunk", minimum=1,
        placeholder="4080s 实测 1024；4090 用 1408",
    ),
    Field(
        "spec", "perf", "投机解码后端（--spec）", kind="enum", cli="--spec", choices=SPECS,
        help="启动期固定；留空 = 不加载投机后端",
    ),
    Field(
        "draft_tokens", "perf", "草稿 token 数（--draft-tokens）", kind="int",
        cli="--draft-tokens", minimum=1, maximum=15,
        show_if={"key": "spec", "nonempty": True},
    ),
    Field(
        "lm_head_draft", "perf", "优化提案头（--lm-head-draft）", kind="bool",
        cli="--lm-head-draft", show_if={"key": "spec", "nonempty": True},
    ),
    Field(
        "ngram", "perf", "n-gram 投机（--ngram）", kind="enum", cli="--ngram", choices=("chain",),
    ),
    Field(
        "ngram_max", "perf", "n-gram 上限（--ngram-max）", kind="int", cli="--ngram-max",
        minimum=1, advanced=True, show_if={"key": "ngram", "values": ["chain"]},
    ),
    Field(
        "ngram_n", "perf", "n-gram 长度（--ngram-n）", kind="int", cli="--ngram-n",
        minimum=1, advanced=True, show_if={"key": "ngram", "values": ["chain"]},
    ),
    Field(
        "ngram_min", "perf", "n-gram 最小匹配（--ngram-min）", kind="int", cli="--ngram-min",
        minimum=1, advanced=True, show_if={"key": "ngram", "values": ["chain"]},
    ),
    Field(
        "ngram_pool_mib", "perf", "n-gram 池 MiB（--ngram-pool-mib）", kind="int",
        cli="--ngram-pool-mib", minimum=0, advanced=True,
        show_if={"key": "ngram", "values": ["chain"]},
    ),
    Field(
        "no_cuda_graph", "perf", "禁用 CUDA Graph（--no-cuda-graph）", kind="bool",
        cli="--no-cuda-graph", advanced=True,
        help="长投机运行偶发崩溃时的规避项",
    ),
    # ---------------------------------------------------------------- 缓存
    Field(
        "device_state_slots", "cache", "设备检查点槽（--device-state-slots）", kind="int",
        cli="--device-state-slots", minimum=0, placeholder="serve 默认 = 并发数",
    ),
    Field(
        "host_state_slots", "cache", "主机状态槽（--host-state-slots）", kind="int",
        cli="--host-state-slots", minimum=0, placeholder="serve 默认 8",
    ),
    Field(
        "host_kv_mib", "cache", "主机 KV 池 MiB（--host-kv-mib）", kind="int",
        cli="--host-kv-mib", minimum=0, placeholder="serve 默认 8192（主机内存，非显存）",
    ),
    Field(
        "no_prefix_reuse", "cache", "禁用前缀复用（--no-prefix-reuse）", kind="bool",
        cli="--no-prefix-reuse",
    ),
    Field(
        "max_private_continuations", "cache", "私有续写上限（--max-private-continuations）",
        kind="int", cli="--max-private-continuations", minimum=0, advanced=True,
        placeholder="serve 默认 2×并发",
    ),
    Field(
        "max_shared_prefixes", "cache", "共享前缀上限（--max-shared-prefixes）", kind="int",
        cli="--max-shared-prefixes", minimum=0, advanced=True,
        placeholder="serve 默认 max(并发,4)",
    ),
    Field(
        "max_long_anchors_per_continuation", "cache",
        "长锚点上限（--max-long-anchors-per-continuation）", kind="int",
        cli="--max-long-anchors-per-continuation", minimum=0, advanced=True,
    ),
    Field(
        "auto_long_anchors", "cache", "自动长锚点（--auto-long-anchors）", kind="int",
        cli="--auto-long-anchors", minimum=0, advanced=True,
        placeholder="留空 = 上限值；0 关闭",
    ),
    Field(
        "slot_save_path", "cache", "会话存档目录（--slot-save-path）", kind="path",
        cli="--slot-save-path", advanced=True,
        help="启用 /slots/{id}?action=save|restore|erase",
    ),
    Field(
        "auto_save_evicted", "cache", "驱逐时自动存档（--auto-save-evicted）", kind="bool",
        cli="--auto-save-evicted", advanced=True,
        show_if={"key": "slot_save_path", "nonempty": True},
    ),
    Field(
        "response_store_max_records", "cache",
        "Responses 记录数上限（--response-store-max-records）", kind="int",
        cli="--response-store-max-records", minimum=1, advanced=True, placeholder="默认 1024",
    ),
    Field(
        "response_store_max_mib", "cache", "Responses 容量 MiB（--response-store-max-mib）",
        kind="int", cli="--response-store-max-mib", minimum=1, advanced=True, placeholder="默认 256",
    ),
    # ---------------------------------------------------------------- 请求默认
    Field(
        "default_max_tokens", "request", "默认输出上限（--default-max-tokens）", kind="int",
        cli="--default-max-tokens", minimum=1, placeholder="serve 默认 8192",
    ),
    Field(
        "default_thinking_budget", "request", "默认思考预算（--default-thinking-budget）",
        kind="int", cli="--default-thinking-budget", minimum=1,
        help="限制模型思考 token；不能与 --no-thinking 同时使用",
    ),
    Field(
        "preserve_thinking", "request", "保留历史思考（--preserve-thinking）", kind="bool",
        cli="--preserve-thinking",
    ),
    Field(
        "no_thinking", "request", "关闭思考（--no-thinking）", kind="bool", cli="--no-thinking",
    ),
    Field(
        "temperature", "request", "temperature（--temperature）", kind="float",
        cli="--temperature", minimum=0, maximum=5, advanced=True,
    ),
    Field(
        "top_p", "request", "top-p（--top-p）", kind="float", cli="--top-p",
        minimum=0, maximum=1, advanced=True,
    ),
    Field(
        "top_k", "request", "top-k（--top-k）", kind="int", cli="--top-k",
        minimum=0, advanced=True,
    ),
    Field(
        "min_p", "request", "min-p（--min-p）", kind="float", cli="--min-p",
        minimum=0, maximum=1, advanced=True,
    ),
    Field(
        "presence_penalty", "request", "存在惩罚（--presence-penalty）", kind="float",
        cli="--presence-penalty", advanced=True,
    ),
    Field(
        "frequency_penalty", "request", "频率惩罚（--frequency-penalty）", kind="float",
        cli="--frequency-penalty", advanced=True,
    ),
    Field(
        "seed", "request", "随机种子（--seed）", kind="int", cli="--seed", advanced=True,
    ),
    Field(
        "greedy", "request", "贪心解码（--greedy）", kind="bool", cli="--greedy", advanced=True,
        help="强制 temperature 0（精确 argmax）",
    ),
    # ---------------------------------------------------------------- 网络/日志/媒体
    Field(
        "host", "network", "监听地址（--host）", cli="--host", default="127.0.0.1",
    ),
    Field(
        "port", "network", "服务端口（--port）", kind="int", cli="--port", default=8080,
        minimum=1, maximum=65535,
    ),
    Field(
        "api_key", "network", "API Key（--api-key）", kind="secret", cli="--api-key",
        advanced=True,
    ),
    Field(
        "cors", "network", "启用 CORS（--cors）", kind="bool", cli="--cors", advanced=True,
    ),
    Field(
        "device", "network", "GPU 序号（--device）", kind="int", cli="--device",
        minimum=0, advanced=True,
    ),
    Field(
        "log_level", "network", "日志级别（--log-level）", kind="enum", cli="--log-level",
        choices=LOG_LEVELS, default="info",
    ),
    Field(
        "log_stats_interval_ms", "network", "吞吐日志间隔 ms（--log-stats-interval-ms）",
        kind="int", cli="--log-stats-interval-ms", minimum=0, advanced=True,
        placeholder="默认 5000；0 关闭",
    ),
    Field(
        "request_log_jsonl", "network", "请求日志文件（--request-log-jsonl）", kind="path",
        cli="--request-log-jsonl", advanced=True,
    ),
    Field(
        "context_cost_presets", "network", "上下文成本预设（--context-cost-presets）",
        kind="path", cli="--context-cost-presets", advanced=True,
    ),
    Field(
        "max_request_mib", "network", "请求体上限 MiB（--max-request-mib）", kind="int",
        cli="--max-request-mib", minimum=1, advanced=True, placeholder="默认 384",
    ),
    Field(
        "media_cache_mib", "network", "媒体缓存 MiB（--media-cache-mib）", kind="int",
        cli="--media-cache-mib", minimum=0, advanced=True, placeholder="默认 1024；0 关闭",
    ),
    Field(
        "media_live_mib", "network", "媒体在工作集 MiB（--media-live-mib）", kind="int",
        cli="--media-live-mib", minimum=0, advanced=True, placeholder="默认 2048",
    ),
    Field(
        "media_preprocess_threads", "network", "媒体预处理线程（--media-preprocess-threads）",
        kind="int", cli="--media-preprocess-threads", minimum=0, advanced=True,
        placeholder="默认 0 = 自动",
    ),
    Field(
        "extra_args", "network", "额外参数（原样追加）", kind="raw", advanced=True,
        placeholder='例如 --no-cuda-graph "E:\\path\\x.json"',
        help="按 Windows 习惯拆分后原样追加到命令行末尾；未知参数会让 serve 报错",
    ),
)

FIELD_BY_KEY: dict[str, Field] = {f.key: f for f in FIELDS}

#: 不是 serve 开关的键（位置参数或原样追加）。
PSEUDO_KEYS = ("engine_binary", "artifact", "extra_args")

_BOOL_TRUE = {"1", "true", "yes", "on", "是"}
_BOOL_FALSE = {"0", "false", "no", "off", "否", ""}


# --------------------------------------------------------------------------- 路径


def launcher_home() -> Path:
    """launcher 的用户级目录（profiles/settings/日志）。"""

    env = os.environ.get("NINFER_LAUNCHER_HOME")
    if env:
        return Path(env).expanduser()
    base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
    return Path(base) / "NInfer-Launcher"


def engine_identity(engine_binary: str | None) -> str:
    """引擎的构建标识（界面角标用），绝不拿上游 VERSION 冒充本 fork 的构建。

    仓库根的 ``VERSION`` 按 ``scripts/package-release-*.ps1`` 的注释就是上游 RTX 3090 的
    标签，因此这里只认三种来源：

    1. 引擎同目录的 ``VERSION``（``dist/.../sm80-rtx4080s/VERSION`` → ``0.6.1-rtx4080s``）；
    2. ``dist/<包>/VERSION``（产品级标签 ``0.6.1-sm89``），且引擎确实位于该 dist 包内；
    3. 都不匹配时退化为构建目录名（``build-4080s``），不臆造版本号。
    """

    if not engine_binary:
        return ""
    path = Path(engine_binary)

    sibling = path.parent / "VERSION"
    if sibling.is_file():
        try:
            text = sibling.read_text(encoding="utf-8").strip()
            if text:
                return text
        except OSError:
            pass

    dist_root = REPO_ROOT / "dist"
    relative = None
    if dist_root.is_dir():
        try:
            relative = path.resolve().relative_to(dist_root.resolve())
        except (ValueError, OSError):
            relative = None
    if relative is not None and len(relative.parts) >= 3:
        package_version = dist_root / relative.parts[0] / "VERSION"
        if package_version.is_file():
            try:
                text = package_version.read_text(encoding="utf-8").strip()
                if text:
                    return text
            except OSError:
                pass

    return path.parent.parent.name or path.parent.name


def default_engine_binary() -> str:
    """按优先级探测本机引擎二进制（NINFER_SERVER 优先，其次仓库内构建产物）。"""

    env = os.environ.get("NINFER_SERVER")
    if env and Path(env).is_file():
        return str(Path(env))

    candidates: list[Path] = [
        REPO_ROOT / "build" / "apps" / "ninfer-serve.exe",
        REPO_ROOT / "build-4080s" / "apps" / "ninfer-serve.exe",
        REPO_ROOT / "build-4090" / "apps" / "ninfer-serve.exe",
    ]
    dist = REPO_ROOT / "dist"
    if dist.is_dir():
        try:
            for package in sorted(dist.glob("ninfer-*")):
                if not package.is_dir():
                    continue
                for variant in sorted(part for part in package.iterdir() if part.is_dir()):
                    candidates.append(variant / "ninfer-serve.exe")
        except OSError:
            pass

    for path in candidates:
        if path.is_file():
            return str(path)
    return str(candidates[1])


def default_models_roots() -> list[str]:
    """常见模型目录：环境变量、仓库内、同盘根目录的 models、用户目录。"""

    candidates: list[Path] = []
    env = os.environ.get("NINFER_MODEL_DIR")
    if env:
        candidates.append(Path(env))
    candidates += [REPO_ROOT / "models", REPO_ROOT / "out"]
    # 仓库所在盘的根目录（例如仓库在 E:\workspaces\..., 模型放在 E:\models）。
    if REPO_ROOT.anchor:
        candidates.append(Path(REPO_ROOT.anchor) / "models")
    candidates.append(Path.home() / "models")

    out: list[str] = []
    seen: set[str] = set()
    for root in candidates:
        try:
            text = str(root)
            key = text.lower()
            if key in seen or not Path(text).is_dir():
                continue
            seen.add(key)
            out.append(text)
        except OSError:
            continue
    return out or [str(REPO_ROOT / "models")]


def scan_models(roots: Iterable[str]) -> list[dict[str, Any]]:
    """列出 roots 下的 .ninfer（含一层子目录，兼容 ``<name>/<name>.ninfer`` 容器）。"""

    found: dict[str, dict[str, Any]] = {}
    for raw_root in roots:
        root = Path(raw_root)
        if not root.is_dir():
            continue
        try:
            paths = list(root.glob("*.ninfer")) + list(root.glob("*/*.ninfer"))
        except OSError:
            continue
        for path in paths:
            try:
                if not path.is_file():
                    continue
                stat = path.stat()
            except OSError:
                continue
            key = str(path.resolve())
            found[key] = {
                "path": str(path),
                "name": path.name,
                "size": stat.st_size,
                "mtime": stat.st_mtime,
                "root": str(root),
            }
    return sorted(found.values(), key=lambda item: item["path"].lower())


def resolve_artifact(name: str, roots: Iterable[str] | None = None) -> str:
    """在模型根目录里解析 ``name``，支持文件与同名容器目录两种布局。"""

    roots = list(roots) if roots else default_models_roots()
    for raw_root in roots:
        root = Path(raw_root)
        for candidate in (root / name, root / name / name):
            if candidate.is_file():
                return str(candidate)
        try:
            for candidate in root.glob(f"*/{name}"):
                if candidate.is_file():
                    return str(candidate)
        except OSError:
            continue
    return str(Path(roots[0]) / name) if roots else name


def expand_artifact(path: str) -> str:
    """目录 → 其中的第一个 ``*.ninfer``（与 start-*.bat 的行为一致）。"""

    if not path:
        return path
    candidate = Path(path)
    if candidate.is_dir():
        try:
            for item in sorted(candidate.glob("*.ninfer")):
                if item.is_file():
                    return str(item)
        except OSError:
            pass
    return path


# ---------------------------------------------------------------- artifact 能力


@dataclass(frozen=True)
class ArtifactInfo:
    """从 v3 artifact 目录里读到的最小能力信息（组件、架构、上下文上限）。"""

    path: str
    ok: bool
    name: str = ""
    components: tuple[str, ...] = ()
    architecture: str = ""
    max_context: int | None = None
    #: ``--lm-head-draft`` 用的是 text 组件里的 proposal 成员，不是顶层组件。
    proposal: bool = False
    error: str = ""

    @property
    def speculative_backends(self) -> tuple[str, ...]:
        return tuple(name for name in ("mtp", "dflash", "dflash2") if name in self.components)

    def to_json(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "ok": self.ok,
            "name": self.name,
            "components": list(self.components),
            "speculative_backends": list(self.speculative_backends),
            "architecture": self.architecture,
            "max_context": self.max_context,
            "proposal": self.proposal,
            "error": self.error,
        }


#: (路径, mtime_ns, 大小) → 能力信息；读目录 JSON 很便宜，但表单预览会频繁调用。
_ARTIFACT_CACHE: dict[tuple[str, int, int], ArtifactInfo] = {}


def inspect_artifact(path: str) -> ArtifactInfo:
    """用项目自带的 artifact 读取器列组件；失败只返回错误，不抛异常。"""

    resolved = expand_artifact(path)
    if not resolved:
        return ArtifactInfo(path="", ok=False, error="未选择模型文件")
    try:
        stat = Path(resolved).stat()
    except OSError as exc:
        return ArtifactInfo(path=resolved, ok=False, error=f"无法访问：{exc}")

    key = (resolved, stat.st_mtime_ns, stat.st_size)
    cached = _ARTIFACT_CACHE.get(key)
    if cached is not None:
        return cached

    try:
        from tools.artifact.reader import Artifact  # 延迟导入：仅在选择模型文件时需要
    except Exception as exc:  # noqa: BLE001 - 读取器缺失不应让 launcher 起不来
        return ArtifactInfo(path=resolved, ok=False, error=f"artifact 读取器不可用：{exc}")

    try:
        artifact = Artifact(resolved)
    except Exception as exc:  # noqa: BLE001 - 非 v3 产物、损坏文件都走这里
        return ArtifactInfo(path=resolved, ok=False, error=str(exc))

    try:
        components = tuple(sorted(artifact.directory.components))
        name = str(artifact.directory.metadata.get("name") or "")
        text_component = artifact.directory.components.get("text") or {}
        text_config = text_component.get("config") or {} if isinstance(text_component, dict) else {}
        proposal = bool(isinstance(text_component, dict) and text_component.get("proposal"))
        architectures = text_config.get("architectures") or []
        architecture = str(architectures[0]) if architectures else ""
        max_context = text_config.get("max_position_embeddings")
        max_context = int(max_context) if isinstance(max_context, (int, float)) else None
    except Exception as exc:  # noqa: BLE001
        return ArtifactInfo(path=resolved, ok=False, error=f"无法解析 artifact 目录：{exc}")
    finally:
        try:
            artifact.close()
        except Exception:  # noqa: BLE001
            pass

    info = ArtifactInfo(
        path=resolved,
        ok=True,
        name=name,
        components=components,
        architecture=architecture,
        max_context=max_context,
        proposal=proposal,
    )
    _ARTIFACT_CACHE[key] = info
    return info


def capability_errors(values: dict[str, Any], info: ArtifactInfo) -> list[str]:
    """按 artifact 实际组件校验 --spec / --vision / --lm-head-draft。

    引擎在加载阶段用 ``directory.component(name)`` 取组件，缺失就抛
    ``missing component <name>``；提前在这里拦住可以省掉一次几十秒的加载失败。
    """

    errors: list[str] = []
    components = info.components
    available = "、".join(components) if components else "无"

    if values.get("vision") and "vision" not in components:
        errors.append(f"该 artifact 不含 vision 组件，不能启用 --vision（组件：{available}）")

    spec = str(values.get("spec") or "")
    if spec and spec not in components:
        backends = info.speculative_backends
        hint = "、".join(backends) if backends else "无"
        errors.append(f"该 artifact 不含 {spec} 组件，不能使用 --spec {spec}（可用投机后端：{hint}）")

    if values.get("lm_head_draft") and not info.proposal:
        errors.append(
            f"该 artifact 的 text 组件没有 proposal 段，不能使用 --lm-head-draft（组件：{available}）"
        )

    return errors


# --------------------------------------------------------------------- 命令行


def split_extra_args(text: str) -> list[str]:
    """拆分「额外参数」：破坏性小、兼容 Windows 路径与引号。"""

    text = (text or "").strip()
    if not text:
        return []
    try:
        tokens = shlex.split(text, posix=False)
    except ValueError:
        tokens = text.split()
    out: list[str] = []
    for token in tokens:
        if len(token) >= 2 and token[0] == '"' and token[-1] == '"':
            token = token[1:-1]
        out.append(token)
    return out


def is_set(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip() != ""
    return True


def format_number(value: float | int) -> str:
    if isinstance(value, int):
        return str(value)
    if float(value).is_integer():
        return str(int(value))
    return f"{value:g}"


#: 自定义参数行：``--flag`` + 可选值。允许字母数字与连字符，避免把奇怪东西塞进 argv。
_CUSTOM_FLAG = re.compile(r"^--[A-Za-z0-9][A-Za-z0-9-]*$")


def normalize_custom_args(raw: Any) -> tuple[list[dict[str, str]], list[str]]:
    """规范「自定义参数」列表；空行忽略，非法名字报错。"""

    if not raw:
        return [], []
    if not isinstance(raw, list):
        return [], ["自定义参数必须是 {flag, value} 列表"]

    rows: list[dict[str, str]] = []
    errors: list[str] = []
    for entry in raw:
        if isinstance(entry, str):
            flag, value = entry.strip(), ""
        elif isinstance(entry, dict):
            flag = str(entry.get("flag") or "").strip()
            value = entry.get("value")
            value = "" if value is None else str(value).strip()
        else:
            errors.append("自定义参数行格式无效")
            continue
        if not flag:
            continue
        if not _CUSTOM_FLAG.match(flag):
            errors.append(f"自定义参数名不合法：{flag}（要写成 --flag）")
            continue
        rows.append({"flag": flag, "value": value})
    return rows, errors


def build_argv(values: dict[str, Any], engine_binary: str | None = None) -> list[str]:
    """参数字典 → 完整 argv（不含 shell 引号）。"""

    binary = engine_binary or str(values.get("engine_binary") or "").strip()
    argv: list[str] = [binary]
    artifact = expand_artifact(str(values.get("artifact") or "").strip())
    if artifact:
        argv.append(artifact)
    for fld in FIELDS:
        if fld.cli is None:
            continue
        value = values.get(fld.key)
        if not is_set(value):
            continue
        if fld.kind == "bool":
            if bool(value):
                argv.append(fld.cli or "")
        elif fld.kind in ("int", "float"):
            try:
                number = int(value) if fld.kind == "int" else float(value)
            except (TypeError, ValueError):
                continue
            argv.extend([fld.cli or "", format_number(number)])
        else:
            argv.extend([fld.cli or "", str(value).strip()])
    # 自定义参数行：原样追加，顺序与界面一致。
    rows, _ = normalize_custom_args(values.get("custom_args"))
    for row in rows:
        argv.append(row["flag"])
        if row["value"]:
            argv.append(row["value"])
    argv.extend(split_extra_args(str(values.get("extra_args") or "")))
    return argv


def display_command(argv: list[str]) -> str:
    return subprocess.list2cmdline(argv)


# ---------------------------------------------------------------------- 校验


def coerce(field_def: Field, raw: Any) -> Any:
    """把浏览器传来的原始值收敛为该字段的类型；失败抛 ValueError。"""

    if field_def.kind == "bool":
        if isinstance(raw, bool):
            return raw
        if raw is None:
            return False
        text = str(raw).strip().lower()
        if text in _BOOL_TRUE:
            return True
        if text in _BOOL_FALSE or text in ("none", "null"):
            return False
        raise ValueError(f"「{field_def.label}」需要是布尔值")
    if field_def.kind == "auto_int":
        text = str(raw).strip() if raw is not None else ""
        if text == "" or text.lower() == "auto":
            return "auto"
        try:
            return int(text)
        except ValueError as exc:
            raise ValueError(f"「{field_def.label}」需要是整数或 auto") from exc
    if not is_set(raw):
        return None
    if field_def.kind == "int":
        try:
            return int(str(raw).strip())
        except ValueError as exc:
            raise ValueError(f"「{field_def.label}」需要是整数") from exc
    if field_def.kind == "float":
        try:
            return float(str(raw).strip())
        except ValueError as exc:
            raise ValueError(f"「{field_def.label}」需要是数字") from exc
    if field_def.kind == "enum":
        text = str(raw).strip()
        if text not in field_def.choices:
            allowed = "、".join(field_def.choices)
            raise ValueError(f"「{field_def.label}」只能是 {allowed}")
        return text
    return str(raw).strip()


def normalize(values: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """收敛所有已知字段；返回 (values, errors)。未知键保留（便于前向兼容但不入命令行）。"""

    errors: list[str] = []
    out: dict[str, Any] = {}
    for fld in FIELDS:
        raw = values.get(fld.key, fld.default)
        try:
            out[fld.key] = coerce(fld, raw)
        except ValueError as exc:
            errors.append(str(exc))
            out[fld.key] = fld.default
    rows, custom_errors = normalize_custom_args(values.get("custom_args"))
    out["custom_args"] = rows
    errors.extend(custom_errors)
    return out, errors


def default_values(engine_binary: str | None = None) -> dict[str, Any]:
    values = {fld.key: fld.default for fld in FIELDS}
    values["engine_binary"] = engine_binary or default_engine_binary()
    values["custom_args"] = []
    return values


def validate(values: dict[str, Any]) -> tuple[list[str], list[str]]:
    """语义校验：返回 (errors, warnings)。errors 阻止启动，warnings 只提示。"""

    errors: list[str] = []
    warnings: list[str] = []

    binary = str(values.get("engine_binary") or "").strip()
    if not binary:
        errors.append("请选择 ninfer-serve.exe（引擎二进制）")
    elif not Path(binary).is_file():
        errors.append(f"引擎二进制不存在：{binary}")

    artifact = expand_artifact(str(values.get("artifact") or "").strip())
    if not artifact:
        errors.append("请选择 .ninfer 模型文件")
    elif not Path(artifact).is_file():
        errors.append(f"模型文件不存在：{artifact}")

    # artifact 能力校验：--spec / --vision / --lm-head-draft 都要求产物里带对应组件，
    # 缺失时引擎会在加载阶段直接拒绝启动，这里提前拦住。
    info = inspect_artifact(artifact) if artifact and Path(artifact).is_file() else None
    if info is not None:
        if info.ok:
            errors.extend(capability_errors(values, info))
        else:
            warnings.append(f"无法读取 artifact 组件，跳过能力校验（{info.error}）")

    def as_int(key: str) -> int | None:
        value = values.get(key)
        if value is None or value == "":
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    concurrency = as_int("max_concurrency")
    if concurrency is not None and not 1 <= concurrency <= 8:
        errors.append("最大并发必须在 1–8 之间（产品上限）")

    context = as_int("max_context")
    if context is not None:
        if context < 1:
            errors.append("最大上下文必须 ≥ 1")
        elif info is not None and info.ok and info.max_context:
            if context > info.max_context:
                errors.append(
                    f"最大上下文 {context} 超过 artifact 声明的上限 {info.max_context}"
                )
        elif context > CONTEXT_CEILING:
            warnings.append(
                f"最大上下文 {context} 超过 {CONTEXT_CEILING}：artifact 常见上限，服务可能拒绝启动"
            )

    kv_capacity = values.get("kv_capacity")
    if isinstance(kv_capacity, int) and kv_capacity < 1:
        errors.append("KV 池容量必须是正整数或 auto")

    spec = values.get("spec") or ""
    draft = as_int("draft_tokens")
    if spec:
        if draft is None:
            errors.append("使用 --spec 时必须填写 --draft-tokens")
        elif spec == "dflash2" and not 1 <= draft <= 15:
            errors.append("DFlash2 的 --draft-tokens 合法范围是 1–15")
    elif draft is not None:
        warnings.append("填写了 --draft-tokens 但未选择 --spec，该参数不会生效")
    if values.get("lm_head_draft") and not spec:
        warnings.append("--lm-head-draft 需要配合 --spec 使用")

    ngram = values.get("ngram") or ""
    ngram_children = ("ngram_max", "ngram_n", "ngram_min", "ngram_pool_mib")
    if any(is_set(values.get(key)) for key in ngram_children) and ngram != "chain":
        warnings.append("n-gram 子参数只在 --ngram chain 时生效")

    if is_set(values.get("vision_max_tokens")) and not values.get("vision"):
        warnings.append("--vision-max-tokens 只在启用 --vision 时生效")

    if values.get("auto_save_evicted") and not is_set(values.get("slot_save_path")):
        errors.append("--auto-save-evicted 需要先设置 --slot-save-path")

    if values.get("no_thinking") and is_set(values.get("default_thinking_budget")):
        errors.append("--default-thinking-budget 不能与 --no-thinking 同时使用")

    if values.get("greedy") and is_set(values.get("temperature")):
        warnings.append("--greedy 会强制 temperature 0，显式 temperature 将被忽略")

    host = str(values.get("host") or "").strip()
    if not host:
        errors.append("监听地址不能为空")
    port = as_int("port")
    if port is not None and not 1 <= port <= 65535:
        errors.append("端口必须在 1–65535 之间")

    for key, label in (
        ("device_state_slots", "设备检查点槽"),
        ("host_state_slots", "主机状态槽"),
        ("host_kv_mib", "主机 KV 池"),
        ("max_pending_requests", "排队上限"),
        ("pending_timeout_ms", "排队超时"),
    ):
        value = as_int(key)
        if value is not None and value < 0:
            errors.append(f"{label}不能为负数")

    # 自定义参数：与表单字段重名会生成两条相同开关，serve 只认最后一条。
    rows, _ = normalize_custom_args(values.get("custom_args"))
    if rows:
        modelled = {fld.cli for fld in FIELDS if fld.cli}
        seen: set[str] = set()
        for row in rows:
            flag = row["flag"]
            if flag in modelled:
                warnings.append(f"自定义参数 {flag} 与表单字段重名，命令行里会出现两次")
            if flag in seen:
                warnings.append(f"自定义参数 {flag} 重复出现")
            seen.add(flag)

    return errors, warnings


# --------------------------------------------------------------------- 预设


def builtin_presets() -> list[dict[str, Any]]:
    """仓库四个 ``start-*.bat`` 的等价档案（值逐字段对照脚本）。"""

    roots = default_models_roots()
    engine_4080s = str(REPO_ROOT / "build-4080s" / "apps" / "ninfer-serve.exe")
    engine_4090 = str(REPO_ROOT / "build-4090" / "apps" / "ninfer-serve.exe")
    if not Path(engine_4080s).is_file():
        engine_4080s = default_engine_binary()
    if not Path(engine_4090).is_file():
        engine_4090 = engine_4080s

    qwen = resolve_artifact("qwen3_8_27b_a8.ninfer", roots)
    bonsai = resolve_artifact("bonsai2_27b_vl_mtp_q4q5.ninfer", roots)
    # 容器目录布局（<name>/<name>.ninfer）在 .bat 里是靠目录参数解析的，这里直接指向文件。
    qwen = qwen if qwen.lower().endswith(".ninfer") else str(Path(qwen) / "qwen3_8_27b_a8.ninfer")
    bonsai = bonsai if bonsai.lower().endswith(".ninfer") else str(Path(bonsai) / "bonsai2_27b_vl_mtp_q4q5.ninfer")

    base = {
        "kv_capacity": "auto",
        "kv_dtype": "rk4v4-e8",
        "host": "127.0.0.1",
        "port": 8080,
        "default_max_tokens": 32768,
    }

    def profile(name: str, engine: str, artifact: str, **overrides: Any) -> dict[str, Any]:
        values = default_values(engine)
        values.update(base)
        values["artifact"] = artifact
        values.update(overrides)
        return {"name": name, "values": values}

    return [
        profile(
            "Qwen3.8-27B int8 · 4080s", engine_4080s, qwen,
            model_id="qwen3.8-27b",
            max_context=262144,
            max_concurrency=2,
            prefill_chunk=1024,
            max_pending_requests=10,
            pending_timeout_ms=600000,
            spec="mtp",
            draft_tokens=3,
            lm_head_draft=True,
            ngram="chain",
            preserve_thinking=True,
            device_state_slots=3,
            host_state_slots=4,
            host_kv_mib=4096,
        ),
        profile(
            "Qwen3.8-27B int8 · 4090", engine_4090, qwen,
            model_id="qwen3.8-27b",
            max_context=100000,
            max_concurrency=3,
            prefill_chunk=1408,
            max_pending_requests=10,
            pending_timeout_ms=600000,
            spec="mtp",
            draft_tokens=3,
            lm_head_draft=True,
            ngram="chain",
            preserve_thinking=True,
            device_state_slots=3,
            host_state_slots=4,
            host_kv_mib=4096,
        ),
        profile(
            "Bonsai2-27B VL · 4080s", engine_4080s, bonsai,
            model_id="bonsai-27b",
            max_context=262144,
            max_concurrency=3,
            prefill_chunk=1024,
            spec="mtp",
            draft_tokens=2,
            lm_head_draft=True,
            ngram="chain",
            vision=True,
        ),
        profile(
            "Bonsai2-27B VL · 4090", engine_4090, bonsai,
            model_id="bonsai-27b",
            max_context=262144,
            max_concurrency=3,
            prefill_chunk=1408,
            spec="mtp",
            draft_tokens=2,
            lm_head_draft=True,
            ngram="chain",
            vision=True,
        ),
    ]


# ---------------------------------------------------------------------- 存储


class Store:
    """profiles 与 launcher 设置的 JSON 持久化（单进程、单写者，足够）。"""

    def __init__(self, home: Path | None = None) -> None:
        self.home = Path(home) if home else launcher_home()
        self.profiles_path = self.home / "profiles.json"
        self.settings_path = self.home / "settings.json"
        self._profiles: dict[str, Any] = {"profiles": {}, "last": None}
        self._settings: dict[str, Any] = {}
        self._load()

    # -- IO ----------------------------------------------------------------
    def _load(self) -> None:
        self.home.mkdir(parents=True, exist_ok=True)
        self._profiles = self._read_json(self.profiles_path, {"profiles": {}, "last": None})
        self._settings = self._read_json(self.settings_path, {})
        self._profiles.setdefault("profiles", {})
        self._profiles.setdefault("last", None)

    @staticmethod
    def _read_json(path: Path, fallback: dict[str, Any]) -> dict[str, Any]:
        try:
            with path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            return data if isinstance(data, dict) else dict(fallback)
        except (OSError, ValueError):
            return dict(fallback)

    def _write_json(self, path: Path, data: dict[str, Any]) -> None:
        self.home.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2)
        os.replace(tmp, path)

    # -- settings ----------------------------------------------------------
    @property
    def settings(self) -> dict[str, Any]:
        settings = dict(self._settings)
        settings.setdefault("ui_port", UI_PORT)
        settings.setdefault("engine_binary", default_engine_binary())
        settings.setdefault("models_roots", default_models_roots())
        settings.setdefault("last_values", {})
        settings.setdefault("hf_endpoint", DEFAULT_HF_ENDPOINT)
        settings.setdefault("hf_proxy", "")
        return settings

    def update_settings(self, patch: dict[str, Any]) -> dict[str, Any]:
        settings = self.settings
        for key in ("ui_port", "engine_binary", "models_roots", "last_values", "hf_endpoint", "hf_proxy"):
            if key in patch:
                settings[key] = patch[key]
        self._settings = settings
        self._write_json(self.settings_path, settings)
        return settings

    # -- profiles ----------------------------------------------------------
    def list_profiles(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for name, entry in sorted(self._profiles["profiles"].items(), key=lambda kv: kv[0].lower()):
            values, errors = normalize(entry.get("values") or {})
            out.append({
                "name": name,
                "values": values,
                "builtin": bool(entry.get("builtin")),
                "updated": entry.get("updated"),
                "errors": errors,
            })
        return out

    def get_profile(self, name: str) -> dict[str, Any] | None:
        entry = self._profiles["profiles"].get(name)
        if entry is None:
            return None
        values, _ = normalize(entry.get("values") or {})
        return {"name": name, "values": values, "builtin": bool(entry.get("builtin"))}

    def save_profile(self, name: str, values: dict[str, Any], builtin: bool = False) -> dict[str, Any]:
        import datetime as _dt

        name = name.strip()
        if not name:
            raise ValueError("档案名不能为空")
        normalized, errors = normalize(values)
        self._profiles["profiles"][name] = {
            "values": normalized,
            "builtin": builtin,
            "updated": _dt.datetime.now().isoformat(timespec="seconds"),
        }
        self._profiles["last"] = name
        self._write_json(self.profiles_path, self._profiles)
        return {"name": name, "values": normalized, "builtin": builtin, "errors": errors}

    def delete_profile(self, name: str) -> bool:
        if name not in self._profiles["profiles"]:
            return False
        del self._profiles["profiles"][name]
        if self._profiles.get("last") == name:
            self._profiles["last"] = None
        self._write_json(self.profiles_path, self._profiles)
        return True

    @property
    def last_profile(self) -> str | None:
        return self._profiles.get("last")

    def import_presets(self, overwrite: bool = False) -> list[str]:
        """导入内置预设；默认只补缺失的。返回新增的名字。"""

        added: list[str] = []
        previous_last = self._profiles.get("last")
        for preset in builtin_presets():
            name = preset["name"]
            if name in self._profiles["profiles"] and not overwrite:
                continue
            self.save_profile(name, preset["values"], builtin=True)
            added.append(name)
        # save_profile 会把 last 改成最后一个预设；首次导入时改为第一个，其余情况保持原值。
        if previous_last is not None:
            self._profiles["last"] = previous_last
        elif added:
            self._profiles["last"] = added[0]
        self._write_json(self.profiles_path, self._profiles)
        return added