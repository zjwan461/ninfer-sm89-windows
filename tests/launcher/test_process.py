"""Launcher 进程监督：日志总线、端口探测、子进程生命周期。"""

from __future__ import annotations

import asyncio
import socket
import sys
from pathlib import Path

import pytest

from tools.launcher.process import LogBus, ServeManager, kill_pid, port_in_use


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def test_logbus_tail_and_clear() -> None:
    bus = LogBus(capacity=3)
    for index in range(5):
        bus.append(f"line-{index}")
    tail = bus.tail(10)
    assert [item["text"] for item in tail] == ["line-2", "line-3", "line-4"]
    assert tail[0]["stream"] == "serve"

    bus.append("事件", stream="launcher")
    assert bus.tail(1)[0]["stream"] == "launcher"

    bus.clear()
    assert bus.tail(5) == []


def test_port_in_use_detects_listener() -> None:
    port = _free_port()
    assert port_in_use("127.0.0.1", port, timeout=0.2) is False
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.bind(("127.0.0.1", port))
        server.listen(1)
        assert port_in_use("127.0.0.1", port, timeout=0.5) is True


def test_kill_pid_refuses_non_serve_process() -> None:
    ok, message = kill_pid(999_999)
    assert ok is False
    assert "不存在" in message


def test_serve_manager_lifecycle(tmp_path: Path) -> None:
    """假子进程：写一行日志、保持存活，验证 starting → 停止 与日志尾随。"""

    async def scenario() -> None:
        bus = LogBus()
        manager = ServeManager(bus, log_dir=tmp_path / "logs")
        port = _free_port()
        script = "import time\nprint('hello-serve', flush=True)\ntime.sleep(30)\n"
        argv = [sys.executable, "-u", "-c", script]

        await manager.start(argv, str(tmp_path), "127.0.0.1", port)
        assert manager.state == "starting"
        assert manager.pid is not None

        for _ in range(60):
            if any("hello-serve" in item["text"] for item in bus.tail(50)):
                break
            await asyncio.sleep(0.1)
        assert any("hello-serve" in item["text"] for item in bus.tail(50)), "子进程日志未送达"
        assert manager.log_path is not None and manager.log_path.is_file()

        snapshot = await manager.stop()
        assert snapshot["pid"] is None
        assert manager.pid is None
        await manager.shutdown()

    asyncio.run(scenario())


def test_check_external_flips_between_states(tmp_path: Path) -> None:
    async def scenario() -> None:
        bus = LogBus()
        manager = ServeManager(bus, log_dir=tmp_path / "logs")
        port = _free_port()

        await manager.check_external("127.0.0.1", port)
        assert manager.state == "stopped"

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
            server.bind(("127.0.0.1", port))
            server.listen(1)
            await manager.check_external("127.0.0.1", port)
            assert manager.state == "external"

        await manager.check_external("127.0.0.1", port)
        assert manager.state == "stopped"
        await manager.shutdown()

    asyncio.run(scenario())


def test_stop_refuses_external_mode(tmp_path: Path) -> None:
    async def scenario() -> None:
        bus = LogBus()
        manager = ServeManager(bus, log_dir=tmp_path / "logs")
        manager.state = "external"
        with pytest.raises(RuntimeError):
            await manager.stop()

    asyncio.run(scenario())