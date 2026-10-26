"""下载管理：URL 校验、完整下载、断点续传与取消。"""

from __future__ import annotations

import asyncio
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from tools.launcher.downloads import DownloadManager

PAYLOAD = bytes(range(256)) * 1200  # 307200 字节


class _Handler(BaseHTTPRequestHandler):
    payload = PAYLOAD
    delay = 0.0

    def do_GET(self) -> None:  # noqa: N802 - 标准库接口
        data = type(self).payload
        start = 0
        header = self.headers.get("Range")
        if header and header.startswith("bytes="):
            start = int(header.split("=", 1)[1].split("-", 1)[0] or 0)
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{len(data) - 1}/{len(data)}")
        else:
            self.send_response(200)
        body = data[start:]
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            for offset in range(0, len(body), 65536):
                self.wfile.write(body[offset:offset + 65536])
                if type(self).delay:
                    time.sleep(type(self).delay)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass

    def log_message(self, *args: object) -> None:  # 静默
        return


def _serve(delay: float = 0.0, payload: bytes | None = None) -> tuple[ThreadingHTTPServer, int]:
    _Handler.delay = delay
    _Handler.payload = PAYLOAD if payload is None else payload
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, server.server_address[1]


async def _wait(
    manager: DownloadManager,
    job_id: str,
    states: tuple[str, ...] = ("done", "failed", "cancelled"),
    timeout: float = 20.0,
) -> dict:
    deadline = time.time() + timeout
    snapshot = next(job for job in manager.snapshot() if job["id"] == job_id)
    while time.time() < deadline:
        snapshot = next(job for job in manager.snapshot() if job["id"] == job_id)
        if snapshot["state"] in states:
            return snapshot
        await asyncio.sleep(0.02)
    return snapshot


def test_start_rejects_unsafe_inputs(tmp_path: Path) -> None:
    async def scenario() -> None:
        manager = DownloadManager()
        for repo, filename, endpoint in [
            ("not-a-repo", "m.ninfer", "https://huggingface.co"),
            ("owner/name", "../escape.ninfer", "https://huggingface.co"),
            ("owner/name", "m.ninfer", "ftp://mirror"),
        ]:
            with pytest.raises(ValueError):
                manager.start(repo, filename, str(tmp_path), endpoint)
        await manager.shutdown()

    asyncio.run(scenario())


def test_start_rejects_bad_proxy(tmp_path: Path) -> None:
    async def scenario() -> None:
        manager = DownloadManager()
        with pytest.raises(ValueError, match="代理"):
            manager.start("owner/name", "m.ninfer", str(tmp_path), "https://huggingface.co", "socks5://127.0.0.1:1080")
        await manager.shutdown()

    asyncio.run(scenario())


def test_download_through_proxy(tmp_path: Path) -> None:
    """代理按 job 透传给 httpx（不发真实请求，只检查 client 构造参数）。"""

    import httpx as httpx_module

    async def scenario() -> None:
        manager = DownloadManager()
        captured: dict[str, object] = {}
        real_client = httpx_module.AsyncClient

        def fake_client(**kwargs: object) -> None:
            captured.update(kwargs)
            raise RuntimeError("stop-after-construct")

        httpx_module.AsyncClient = fake_client  # type: ignore[assignment]
        try:
            job = manager.start(
                "owner/name", "m.ninfer", str(tmp_path),
                "https://huggingface.co", "http://127.0.0.1:7897",
            )
            final = await _wait(manager, job["id"])
        finally:
            httpx_module.AsyncClient = real_client  # type: ignore[assignment]

        assert final["proxy"] == "http://127.0.0.1:7897"
        assert captured.get("proxy") == "http://127.0.0.1:7897"
        assert captured.get("follow_redirects") is True
        await manager.shutdown()

    asyncio.run(scenario())


def test_download_without_proxy_omits_client_proxy(tmp_path: Path) -> None:
    import httpx as httpx_module

    async def scenario() -> None:
        manager = DownloadManager()
        captured: dict[str, object] = {}
        real_client = httpx_module.AsyncClient

        def fake_client(**kwargs: object) -> None:
            captured.update(kwargs)
            raise RuntimeError("stop-after-construct")

        httpx_module.AsyncClient = fake_client  # type: ignore[assignment]
        try:
            job = manager.start("owner/name", "m.ninfer", str(tmp_path), "https://huggingface.co")
            await _wait(manager, job["id"])
        finally:
            httpx_module.AsyncClient = real_client  # type: ignore[assignment]

        # 不显式传 proxy：httpx 的 trust_env 会沿用 HTTPS_PROXY 环境变量。
        assert "proxy" not in captured
        await manager.shutdown()

    asyncio.run(scenario())


def test_download_completes(tmp_path: Path) -> None:
    server, port = _serve()
    try:
        async def scenario() -> None:
            manager = DownloadManager()
            job = manager.start("owner/name", "m.ninfer", str(tmp_path), f"http://127.0.0.1:{port}")
            final = await _wait(manager, job["id"])
            assert final["state"] == "done", final
            assert final["done"] == len(PAYLOAD)
            assert final["percent"] == pytest.approx(100.0)
            assert (tmp_path / "m.ninfer").read_bytes() == PAYLOAD
            await manager.shutdown()

        asyncio.run(scenario())
    finally:
        server.shutdown()


def test_download_resumes_from_partial_file(tmp_path: Path) -> None:
    server, port = _serve()
    try:
        async def scenario() -> None:
            partial = tmp_path / "m.ninfer"
            partial.write_bytes(PAYLOAD[:100_000])
            manager = DownloadManager()
            job = manager.start("owner/name", "m.ninfer", str(tmp_path), f"http://127.0.0.1:{port}")
            final = await _wait(manager, job["id"])
            assert final["state"] == "done", final
            assert partial.read_bytes() == PAYLOAD
            await manager.shutdown()

        asyncio.run(scenario())
    finally:
        server.shutdown()


def test_download_cancel_keeps_partial(tmp_path: Path) -> None:
    big = PAYLOAD * 6  # ~1.8 MB：分块发送足够慢，留出取消窗口
    server, port = _serve(delay=0.05, payload=big)
    try:
        async def scenario() -> None:
            manager = DownloadManager()
            job = manager.start("owner/name", "m.ninfer", str(tmp_path), f"http://127.0.0.1:{port}")
            await asyncio.sleep(0.3)
            assert manager.cancel(job["id"]) is True
            final = await _wait(manager, job["id"])
            assert final["state"] == "cancelled", final
            assert (tmp_path / "m.ninfer").stat().st_size < len(big)
            assert manager.cancel(job["id"]) is False  # 已结束
            assert manager.clear(job["id"]) is True
            assert manager.snapshot() == []
            await manager.shutdown()

        asyncio.run(scenario())
    finally:
        server.shutdown()


def test_clear_rejects_running_job(tmp_path: Path) -> None:
    big = PAYLOAD * 6
    server, port = _serve(delay=0.05, payload=big)
    try:
        async def scenario() -> None:
            manager = DownloadManager()
            job = manager.start("owner/name", "m.ninfer", str(tmp_path), f"http://127.0.0.1:{port}")
            await asyncio.sleep(0.1)
            assert manager.clear(job["id"]) is False
            manager.cancel(job["id"])
            await _wait(manager, job["id"])
            await manager.shutdown()

        asyncio.run(scenario())
    finally:
        server.shutdown()


def test_pause_then_resume_continues_from_offset(tmp_path: Path) -> None:
    """暂停保留已下载字节；继续时用 Range 从断点续传，最终内容与源完全一致。"""

    big = PAYLOAD * 8
    server, port = _serve(delay=0.05, payload=big)
    try:
        async def scenario() -> None:
            manager = DownloadManager()
            job = manager.start("owner/name", "m.ninfer", str(tmp_path), f"http://127.0.0.1:{port}")

            # 等到确实写入了若干字节再暂停，否则读循环可能在第一个分块前就停下。
            deadline = time.time() + 10
            while time.time() < deadline:
                snap = next(item for item in manager.snapshot() if item["id"] == job["id"])
                if snap["done"] > 0:
                    break
                await asyncio.sleep(0.02)
            assert snap["done"] > 0, snap

            assert manager.pause(job["id"]) is True
            paused = await _wait(manager, job["id"], ("paused",))
            assert paused["state"] == "paused", paused
            partial = (tmp_path / "m.ninfer").stat().st_size
            assert 0 < partial < len(big), (partial, len(big))
            assert manager.pause(job["id"]) is False  # 已暂停

            # 暂停状态下也能取消
            assert await manager.resume(job["id"]) is True
            final = await _wait(manager, job["id"])
            assert final["state"] == "done", final
            assert (tmp_path / "m.ninfer").read_bytes() == big
            assert await manager.resume(job["id"]) is False  # 已完成
            await manager.shutdown()

        asyncio.run(scenario())
    finally:
        server.shutdown()


def test_resume_after_cancel_keeps_partial(tmp_path: Path) -> None:
    big = PAYLOAD * 8
    server, port = _serve(delay=0.05, payload=big)
    try:
        async def scenario() -> None:
            manager = DownloadManager()
            job = manager.start("owner/name", "m.ninfer", str(tmp_path), f"http://127.0.0.1:{port}")
            await asyncio.sleep(0.2)
            assert manager.cancel(job["id"]) is True
            cancelled = await _wait(manager, job["id"])
            assert cancelled["state"] == "cancelled"
            partial = (tmp_path / "m.ninfer").stat().st_size

            assert await manager.resume(job["id"]) is True
            final = await _wait(manager, job["id"])
            assert final["state"] == "done", final
            assert final["done"] == len(big)
            assert (tmp_path / "m.ninfer").read_bytes() == big
            assert partial < len(big)
            await manager.shutdown()

        asyncio.run(scenario())
    finally:
        server.shutdown()


def test_pause_requires_active_job(tmp_path: Path) -> None:
    async def scenario() -> None:
        manager = DownloadManager()
        assert manager.pause("nope") is False
        assert await manager.resume("nope") is False
        await manager.shutdown()

    asyncio.run(scenario())