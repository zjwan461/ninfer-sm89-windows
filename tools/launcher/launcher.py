"""NInfer Launcher：本地 WebUI + serve 进程监督（FastAPI 应用）。

启动方式（由仓库根目录的 start-launcher.bat 调用）::

    python tools/launcher/launcher.py [--port 8100] [--no-browser]

只绑定 127.0.0.1：这是单机、单属主的本地工具。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import threading
import webbrowser
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# 允许 ``python tools/launcher/launcher.py`` 与 ``python -m tools.launcher.launcher`` 两种入口。
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.launcher import config  # noqa: E402
from tools.launcher.process import GpuMonitor, LogBus, ServeManager, kill_pid, port_in_use  # noqa: E402

STATIC_DIR = Path(__file__).resolve().parent / "static"


# --------------------------------------------------------------------- 请求模型


class ValuesBody(BaseModel):
    values: dict[str, Any] = Field(default_factory=dict)


class ProfileBody(BaseModel):
    name: str
    values: dict[str, Any] = Field(default_factory=dict)


class SettingsBody(BaseModel):
    ui_port: int | None = None
    engine_binary: str | None = None
    models_roots: list[str] | None = None


class ChatBody(BaseModel):
    prompt: str
    max_tokens: int = 256
    temperature: float | None = None
    system: str | None = None


class KillExternalBody(BaseModel):
    pid: int


# ----------------------------------------------------------------------- 应用


def build_app(store: config.Store, ui_port: int) -> FastAPI:
    logbus = LogBus()
    manager = ServeManager(logbus, log_dir=store.home / "logs")
    gpu = GpuMonitor()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        loop = asyncio.get_running_loop()
        logbus.bind_loop(loop)
        manager.bind_loop(loop)
        gpu.start()
        # 首次运行：导入内置预设。
        if not store.list_profiles():
            added = store.import_presets()
            if added:
                logbus.append(f"已导入 {len(added)} 个内置预设", "launcher")

        last = store.settings.get("last_values") or {}
        host = str(last.get("host") or "127.0.0.1")
        port = int(last.get("port") or 8080)
        await manager.check_external(host, port)
        logbus.append(f"NInfer Launcher 启动（UI {ui_port}，serve 端口 {port}）", "launcher")
        try:
            yield
        finally:
            await gpu.stop()
            await manager.shutdown()

    app = FastAPI(title="NInfer Launcher", lifespan=lifespan)
    app.state.logbus = logbus
    app.state.manager = manager
    app.state.gpu = gpu
    app.state.store = store

    # ------------------------------------------------------------- 静态页面
    # 本地工具：页面资源一律禁止缓存，且资源 URL 带修改时间戳，改完代码刷新即生效。
    @app.middleware("http")
    async def no_cache_static(request: Request, call_next):  # type: ignore[no-untyped-def]
        response = await call_next(request)
        path = request.url.path
        if path == "/" or path.startswith("/static"):
            response.headers["Cache-Control"] = "no-cache, must-revalidate"
        return response

    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/")
    async def index() -> HTMLResponse:
        """index.html 里的 asset URL 附上 mtime，旧缓存不会继续生效。"""

        html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
        stamp = 0
        for asset in ("app.js", "styles.css"):
            try:
                stamp = max(stamp, int((STATIC_DIR / asset).stat().st_mtime))
            except OSError:
                pass
        html = html.replace("/static/app.js", f"/static/app.js?v={stamp}")
        html = html.replace("/static/styles.css", f"/static/styles.css?v={stamp}")
        return HTMLResponse(html, headers={"Cache-Control": "no-cache, must-revalidate"})

    # ------------------------------------------------------------ 元信息
    @app.get("/api/schema")
    async def schema() -> dict[str, Any]:
        settings = store.settings
        return {
            # 引擎构建标识（不是仓库根的 VERSION：那是上游 RTX 3090 的标签）。
            "build": config.engine_identity(settings["engine_binary"]),
            "ui_port": settings["ui_port"],
            "groups": [{"id": gid, "title": title} for gid, title in config.GROUPS],
            "fields": [fld.to_json() for fld in config.FIELDS],
            "defaults": config.default_values(settings["engine_binary"]),
            "models_roots": settings["models_roots"],
            "context_ceiling": config.CONTEXT_CEILING,
            "presets": [preset["name"] for preset in config.builtin_presets()],
        }

    @app.get("/api/settings")
    async def get_settings() -> dict[str, Any]:
        return store.settings

    @app.post("/api/settings")
    async def post_settings(body: SettingsBody) -> dict[str, Any]:
        patch = body.model_dump(exclude_none=True)
        return store.update_settings(patch)

    @app.get("/api/models")
    async def models(roots: str | None = Query(default=None)) -> dict[str, Any]:
        settings = store.settings
        root_list = [item for item in (roots or ",").split(",") if item.strip()] if roots else settings["models_roots"]
        return {"roots": root_list, "models": config.scan_models(root_list)}

    @app.get("/api/artifact")
    async def artifact(path: str = Query(...)) -> dict[str, Any]:
        """artifact 的组件/架构/上下文上限，供表单提示与能力校验。"""

        info = await asyncio.to_thread(config.inspect_artifact, path)
        return info.to_json()

    # ------------------------------------------------------------- 命令行预览
    @app.post("/api/preview")
    async def preview(body: ValuesBody) -> dict[str, Any]:
        values, errors = config.normalize(body.values)
        semantic_errors, warnings = config.validate(values)
        argv = config.build_argv(values)
        return {
            "values": values,
            "argv": argv,
            "display": config.display_command(argv),
            "errors": errors + semantic_errors,
            "warnings": warnings,
        }

    # ------------------------------------------------------------- 进程
    @app.get("/api/status")
    async def status() -> dict[str, Any]:
        snapshot = manager.snapshot()
        snapshot["gpu"] = gpu.snapshot()
        snapshot["ui_port"] = store.settings["ui_port"]
        engine = snapshot["argv"][0] if snapshot["argv"] else store.settings["engine_binary"]
        snapshot["build"] = config.engine_identity(engine)
        return snapshot

    @app.post("/api/start")
    async def start(body: ValuesBody) -> dict[str, Any]:
        values, errors = config.normalize(body.values)
        semantic_errors, warnings = config.validate(values)
        if errors or semantic_errors:
            raise HTTPException(status_code=400, detail={"errors": errors + semantic_errors, "warnings": warnings})
        if manager.alive():
            raise HTTPException(status_code=409, detail={"errors": ["serve 已经在运行，请先停止"]})

        host = str(values.get("host") or "127.0.0.1")
        port = int(values.get("port") or 8080)
        started = await asyncio.to_thread(port_in_use, host, port, 0.5)
        if started:
            await manager.check_external(host, port)
            pid = manager.external_pid
            raise HTTPException(
                status_code=409,
                detail={
                    "errors": [
                        f"端口 {port} 已被占用（PID {pid}），请先停止该实例或换端口"
                        if pid else f"端口 {port} 已被占用，请换端口"
                    ],
                    "warnings": warnings,
                },
            )

        argv = config.build_argv(values)
        engine_dir = str(Path(argv[0]).resolve().parent)
        store.update_settings({"last_values": values})
        snapshot = await manager.start(argv, engine_dir, host, port)
        snapshot["warnings"] = warnings
        return snapshot

    @app.post("/api/stop")
    async def stop() -> dict[str, Any]:
        try:
            return await manager.stop()
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail={"errors": [str(exc)]}) from exc

    @app.post("/api/kill-external")
    async def kill_external(body: KillExternalBody) -> dict[str, Any]:
        ok, message = await asyncio.to_thread(kill_pid, body.pid)
        logbus.append(message, "launcher")
        if not ok:
            raise HTTPException(status_code=400, detail={"errors": [message]})
        await asyncio.sleep(0.5)
        last = store.settings.get("last_values") or {}
        await manager.check_external(str(last.get("host") or "127.0.0.1"), int(last.get("port") or 8080))
        return {"ok": True, "message": message}

    # ------------------------------------------------------------- 日志
    @app.get("/api/logs")
    async def logs(request: Request, history: int = Query(default=300, ge=0, le=4096)) -> StreamingResponse:
        async def event_stream() -> AsyncIterator[bytes]:
            queue = logbus.subscribe()
            try:
                for line in logbus.tail(history):
                    yield _sse(line)
                while True:
                    if await request.is_disconnected():
                        break
                    try:
                        item = await asyncio.wait_for(queue.get(), timeout=15.0)
                    except asyncio.TimeoutError:
                        yield b": ping\n\n"
                        continue
                    yield _sse(item)
            finally:
                logbus.unsubscribe(queue)

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
        )

    @app.post("/api/logs/clear")
    async def clear_logs() -> dict[str, Any]:
        logbus.clear()
        return {"ok": True}

    # ------------------------------------------------------------- 观测代理
    @app.get("/api/monitor")
    async def monitor() -> JSONResponse:
        return await _serve_json(manager, "/monitor/stats")

    @app.get("/api/proxy/health")
    async def proxy_health() -> JSONResponse:
        return await _serve_json(manager, "/health")

    @app.get("/api/proxy/models")
    async def proxy_models() -> JSONResponse:
        return await _serve_json(manager, "/v1/models")

    # ------------------------------------------------------------- 测试请求
    @app.post("/api/test/chat")
    async def test_chat(body: ChatBody) -> StreamingResponse:
        if manager.state != "ready":
            raise HTTPException(status_code=409, detail={"errors": ["serve 尚未就绪，无法测试"]})
        model_id = await _resolve_model_id(manager)
        messages: list[dict[str, Any]] = []
        if body.system:
            messages.append({"role": "system", "content": body.system})
        messages.append({"role": "user", "content": body.prompt})
        payload: dict[str, Any] = {
            "model": model_id,
            "messages": messages,
            "stream": True,
            "max_tokens": body.max_tokens,
        }
        if body.temperature is not None and body.temperature >= 0:
            payload["temperature"] = body.temperature
        url = f"http://{manager.host}:{manager.port}/v1/chat/completions"

        async def relay() -> AsyncIterator[bytes]:
            import httpx

            try:
                async with httpx.AsyncClient(timeout=None) as client:
                    async with client.stream("POST", url, json=payload) as response:
                        if response.status_code != 200:
                            detail = (await response.aread()).decode("utf-8", "replace")
                            yield _sse({"error": f"HTTP {response.status_code}: {detail[:2000]}"})
                            yield b"data: [DONE]\n\n"
                            return
                        async for chunk in response.aiter_bytes():
                            yield chunk
            except Exception as exc:  # noqa: BLE001
                yield _sse({"error": f"请求失败：{exc}"})
                yield b"data: [DONE]\n\n"

        return StreamingResponse(relay(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache"})

    # ------------------------------------------------------------- 档案
    @app.get("/api/profiles")
    async def profiles() -> dict[str, Any]:
        return {"profiles": store.list_profiles(), "last": store.last_profile}

    @app.post("/api/profiles")
    async def save_profile(body: ProfileBody) -> dict[str, Any]:
        try:
            return store.save_profile(body.name, body.values)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail={"errors": [str(exc)]}) from exc

    @app.delete("/api/profiles")
    async def delete_profile(name: str = Query(...)) -> dict[str, Any]:
        if not store.delete_profile(name):
            raise HTTPException(status_code=404, detail={"errors": [f"档案不存在：{name}"]})
        return {"ok": True}

    @app.post("/api/profiles/presets")
    async def import_presets(overwrite: bool = Query(default=False)) -> dict[str, Any]:
        added = store.import_presets(overwrite=overwrite)
        return {"added": added, "profiles": store.list_profiles()}

    @app.get("/api/profiles/bat")
    async def export_bat(name: str = Query(...)) -> PlainTextResponse:
        entry = store.get_profile(name)
        if entry is None:
            raise HTTPException(status_code=404, detail={"errors": [f"档案不存在：{name}"]})
        text = render_bat(name, entry["values"])
        return PlainTextResponse(text, media_type="text/plain; charset=utf-8",
                                 headers={"Content-Disposition": f'attachment; filename="{_safe_filename(name)}.bat"'})

    return app


# ------------------------------------------------------------------- 工具函数


def _sse(item: dict[str, Any]) -> bytes:
    return b"data: " + json.dumps(item, ensure_ascii=False).encode("utf-8") + b"\n\n"


async def _serve_json(manager: ServeManager, path: str) -> JSONResponse:
    import httpx

    if manager.state not in ("ready", "starting", "external"):
        return JSONResponse({"ok": False, "error": "serve 未运行"})
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            response = await client.get(f"http://{manager.host}:{manager.port}{path}")
        try:
            return JSONResponse(response.json())
        except ValueError:
            return JSONResponse({"ok": False, "error": f"非 JSON 响应（HTTP {response.status_code}）"})
    except Exception as exc:  # noqa: BLE001
        return JSONResponse({"ok": False, "error": str(exc)})


async def _resolve_model_id(manager: ServeManager) -> str:
    """用 /v1/models 返回的别名；取不到时退回 artifact 名。"""

    import httpx

    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            response = await client.get(f"http://{manager.host}:{manager.port}/v1/models")
        data = response.json()
        items = data.get("data") or []
        if items and isinstance(items[0], dict) and items[0].get("id"):
            return str(items[0]["id"])
    except Exception:  # noqa: BLE001
        pass
    return "default"


def _safe_filename(name: str) -> str:
    cleaned = "".join(
        ch if (ch.isascii() and (ch.isalnum() or ch in "-_.")) else "_" for ch in name
    ).strip("_")
    return cleaned or "profile"


def _cmd_escape(value: str) -> str:
    """cmd.exe 批处理里 ``%`` 需要写成 ``%%``。"""

    return value.replace("%", "%%")


def render_bat(name: str, values: dict[str, Any]) -> str:
    """把档案渲染成与仓库 start-*.bat 同形的可执行脚本。"""

    argv = config.build_argv(values)
    engine = _cmd_escape(argv[0])
    artifact = _cmd_escape(argv[1]) if len(argv) > 1 else ""
    rest = [_cmd_escape(part) for part in argv[2:]]
    # 生成的 .bat 保持纯 ASCII（与仓库其他脚本一致），档案名里的非 ASCII 字符换成 ?。
    note = name.encode("ascii", "replace").decode("ascii")
    lines = [
        "@echo off",
        "setlocal",
        f"rem Generated by NInfer Launcher from profile: {note}",
        "rem Append extra ninfer-serve arguments after the script; they are forwarded via %*.",
        f'set "NINFER_SERVER={engine}"',
        "",
        'if not exist "%NINFER_SERVER%" (',
        "  echo Missing %NINFER_SERVER%",
        "  pause",
        "  exit /b 1",
        ")",
        "",
    ]
    head = f'"%NINFER_SERVER%" "{artifact}"'
    if not rest:
        lines.append(head + ' %*')
    else:
        lines.append(head + " ^")
        groups: list[list[str]] = []
        for token in rest:
            if token.startswith("--") or not groups:
                groups.append([token])
            else:
                groups[-1].append(token)
        for index, group in enumerate(groups):
            suffix = ' %*' if index == len(groups) - 1 else " ^"
            lines.append("  " + " ".join(group) + suffix)
    lines.append("pause")
    return "\r\n".join(lines) + "\r\n"


# --------------------------------------------------------------------- 入口


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="NInfer Launcher（本地 WebUI 启动器）")
    parser.add_argument("--port", type=int, default=None, help=f"UI 端口（默认 {config.UI_PORT}）")
    parser.add_argument("--no-browser", action="store_true", help="启动后不自动打开浏览器")
    parser.add_argument("--home", default=None, help="launcher 数据目录（默认 %%LOCALAPPDATA%%\\NInfer-Launcher）")
    args = parser.parse_args(argv)

    store = config.Store(Path(args.home) if args.home else None)
    settings = store.settings
    port = args.port or int(settings.get("ui_port") or config.UI_PORT)

    # 单实例：端口已被占用说明已有 launcher 在跑，只打开浏览器。
    if port_in_use("127.0.0.1", port, timeout=0.5):
        url = f"http://127.0.0.1:{port}"
        print(f"launcher 已在 {url} 运行，直接打开界面。")
        if not args.no_browser:
            webbrowser.open(url)
        return 0

    if port != settings.get("ui_port"):
        store.update_settings({"ui_port": port})

    app = build_app(store, port)
    url = f"http://127.0.0.1:{port}"
    print(f"NInfer Launcher  {url}")
    print(f"  数据目录: {store.home}")
    print(f"  serve 引擎: {settings.get('engine_binary')}")

    if not args.no_browser:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()

    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning", access_log=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())