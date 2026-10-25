"""serve 子进程监督：启动/停止/状态机、日志尾随、健康轮询、外部实例探测。

设计要点：

- 子进程 stdout/stderr 直接重定向到 ``logs/serve-<时间戳>.log``，而不是管道。
  launcher 意外退出时 serve 不会因断管报错，日志也天然落盘；
- launcher 侧用尾随线程读取该文件并推给 ``LogBus``；
- 子进程退出等待、健康探测都在独立线程/任务里完成，不阻塞 HTTP 请求。
"""

from __future__ import annotations

import asyncio
import collections
import csv
import io
import os
import socket
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

CREATE_NO_WINDOW = 0x08000000
CREATE_NEW_PROCESS_GROUP = 0x00000200

#: 子进程退出后等待收尾的最长时间。
KILL_GRACE_SECONDS = 10.0
#: serve 加载 20GB artifact 可能几十秒，ready 前每秒探测一次。
STARTING_POLL_SECONDS = 1.0
READY_POLL_SECONDS = 5.0
#: 尾随线程的轮询间隔。
TAIL_POLL_SECONDS = 0.15


# --------------------------------------------------------------------- 日志


class LogBus:
    """环形缓冲 + 订阅者广播（serve 输出与 launcher 事件共用一条流）。"""

    def __init__(self, capacity: int = 4096) -> None:
        self._lines: collections.deque[dict[str, Any]] = collections.deque(maxlen=capacity)
        self._lock = threading.Lock()
        self._subscribers: list[asyncio.Queue[dict[str, Any]]] = []
        self._loop: asyncio.AbstractEventLoop | None = None

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def append(self, text: str, stream: str = "serve") -> dict[str, Any]:
        line = {"ts": time.time(), "stream": stream, "text": text.rstrip("\r\n")}
        with self._lock:
            self._lines.append(line)
            subscribers = list(self._subscribers)
        if subscribers and self._loop is not None:
            for queue in subscribers:
                self._loop.call_soon_threadsafe(_safe_put, queue, line)
        return line

    def tail(self, count: int) -> list[dict[str, Any]]:
        if count <= 0:
            return []
        with self._lock:
            return list(self._lines)[-count:]

    def subscribe(self) -> asyncio.Queue[dict[str, Any]]:
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=4096)
        with self._lock:
            self._subscribers.append(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue[dict[str, Any]]) -> None:
        with self._lock:
            if queue in self._subscribers:
                self._subscribers.remove(queue)

    def clear(self) -> None:
        with self._lock:
            self._lines.clear()


def _safe_put(queue: asyncio.Queue[dict[str, Any]], item: dict[str, Any]) -> None:
    try:
        queue.put_nowait(item)
    except asyncio.QueueFull:
        pass


# ------------------------------------------------------------------ 端口/进程


def port_in_use(host: str, port: int, timeout: float = 0.4) -> bool:
    target = "127.0.0.1" if host in ("", "0.0.0.0", "::") else host
    try:
        with socket.create_connection((target, port), timeout=timeout):
            return True
    except OSError:
        return False


def _decode(data: bytes) -> str:
    """中文 Windows 的命令输出是 GBK；优先 UTF-8，失败再退回 GBK/mbcs。"""

    for encoding in ("utf-8", "gbk", "mbcs"):
        try:
            return data.decode(encoding)
        except (UnicodeDecodeError, LookupError):
            continue
    return data.decode("utf-8", "replace")


def _run_bytes(cmd: list[str], timeout: float = 10.0) -> subprocess.CompletedProcess[bytes]:
    """捕获原始字节：中文 Windows 的 netstat/tasklist 输出不是 UTF-8，不能走 text 模式。"""

    return subprocess.run(
        cmd, capture_output=True, timeout=timeout, creationflags=CREATE_NO_WINDOW,
    )


def listening_pid(port: int) -> int | None:
    """用 netstat 找监听该端口的 PID（避免引入 psutil 依赖）。"""

    try:
        completed = _run_bytes(["netstat", "-ano"])
    except (OSError, subprocess.SubprocessError):
        return None
    needle = f":{port}"
    for raw in _decode(completed.stdout).splitlines():
        parts = raw.split()
        if len(parts) < 5 or parts[0].upper() != "TCP" or parts[3].upper() != "LISTENING":
            continue
        if parts[1].endswith(needle):
            try:
                return int(parts[4])
            except ValueError:
                continue
    return None


def process_image_name(pid: int) -> str | None:
    try:
        completed = _run_bytes(["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"])
    except (OSError, subprocess.SubprocessError):
        return None
    for row in csv.reader(io.StringIO(_decode(completed.stdout))):
        # 无匹配进程时 tasklist 输出本地化的 "信息: ..." 单字段行，只有真进程行才带 .exe 与 PID 列。
        if len(row) >= 2 and row[0].lower().endswith((".exe", ".com", ".bat")):
            return row[0]
    return None


def kill_pid(pid: int) -> tuple[bool, str]:
    """只在确认目标确实是 ninfer-serve.exe 时结束它。"""

    name = process_image_name(pid)
    if not name:
        return False, f"PID {pid} 不存在"
    if not name.lower().startswith("ninfer-serve"):
        return False, f"PID {pid} 是 {name}，不是 ninfer-serve，拒绝结束"
    try:
        completed = _run_bytes(["taskkill", "/PID", str(pid), "/F"], timeout=15.0)
    except (OSError, subprocess.SubprocessError) as exc:
        return False, f"taskkill 失败：{exc}"
    if completed.returncode != 0:
        detail = _decode(completed.stderr or completed.stdout).strip()
        return False, detail or "taskkill 返回非零"
    return True, f"已结束 PID {pid}（{name}）"


# ------------------------------------------------------------------- GPU


class GpuMonitor:
    """每 2 秒跑一次 nvidia-smi，给仪表盘提供显存读数。"""

    def __init__(self, interval: float = 2.0) -> None:
        self.interval = interval
        self._snapshot: dict[str, Any] = {"ok": False, "error": "尚未采集", "ts": 0.0}
        self._exe: str | None = None
        self._task: asyncio.Task[None] | None = None

    def snapshot(self) -> dict[str, Any]:
        return dict(self._snapshot)

    def _resolve_exe(self) -> str | None:
        if self._exe:
            return self._exe
        system32 = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "nvidia-smi.exe"
        for candidate in ("nvidia-smi", str(system32)):
            try:
                _run_bytes([candidate, "--version"])
                self._exe = candidate
                return candidate
            except (OSError, subprocess.SubprocessError):
                continue
        return None

    def _read(self) -> dict[str, Any]:
        exe = self._resolve_exe()
        if not exe:
            return {"ok": False, "error": "找不到 nvidia-smi", "ts": time.time()}
        query = "name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu"
        try:
            completed = _run_bytes(
                [exe, f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            )
        except (OSError, subprocess.SubprocessError) as exc:
            return {"ok": False, "error": f"nvidia-smi 调用失败：{exc}", "ts": time.time()}
        text = _decode(completed.stdout).strip()
        parts = [part.strip() for part in text.splitlines()[0].split(",")] if text else []
        if len(parts) < 6:
            return {"ok": False, "error": "nvidia-smi 输出无法解析", "ts": time.time()}
        try:
            return {
                "ok": True,
                "name": parts[0],
                "total_mib": float(parts[1]),
                "used_mib": float(parts[2]),
                "free_mib": float(parts[3]),
                "util_percent": float(parts[4]) if parts[4] not in ("", "[N/A]") else None,
                "temp_c": float(parts[5]) if parts[5] not in ("", "[N/A]") else None,
                "ts": time.time(),
            }
        except ValueError:
            return {"ok": False, "error": "nvidia-smi 数值无法解析", "ts": time.time()}

    async def run(self) -> None:
        while True:
            self._snapshot = await asyncio.to_thread(self._read)
            await asyncio.sleep(self.interval)

    def start(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self.run())

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None


# ------------------------------------------------------------- serve 管理


class ServeManager:
    """一个 launcher 至多监督一个 serve 子进程。"""

    def __init__(self, logbus: LogBus, log_dir: Path | None = None) -> None:
        self.logbus = logbus
        self.log_dir = Path(log_dir) if log_dir else None
        self.log_path: Path | None = None
        self._proc: subprocess.Popen[bytes] | None = None
        self._lock = threading.Lock()
        self.state = "stopped"  # stopped | starting | ready | crashed | external
        self.argv: list[str] = []
        self.started_at: float | None = None
        self.ready_at: float | None = None
        self.exit_code: int | None = None
        self.health: dict[str, Any] = {"ok": False, "ts": 0.0}
        self.external_pid: int | None = None
        self._health_task: asyncio.Task[None] | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._host = "127.0.0.1"
        self._port = 8080
        #: 用户主动停止时 Windows 的 TerminateProcess 退出码非 0，不能据此判成崩溃。
        self._stopping = False

    # -- 状态 --------------------------------------------------------------
    @property
    def pid(self) -> int | None:
        proc = self._proc
        return proc.pid if proc is not None and proc.poll() is None else None

    def alive(self) -> bool:
        return self.pid is not None

    @property
    def host(self) -> str:
        return self._host

    @property
    def port(self) -> int:
        return self._port

    def bind(self, host: str, port: int) -> None:
        self._host = host or "127.0.0.1"
        self._port = int(port)

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def snapshot(self) -> dict[str, Any]:
        now = time.time()
        return {
            "state": self.state,
            "pid": self.pid,
            "external_pid": self.external_pid,
            "exit_code": self.exit_code,
            "argv": list(self.argv),
            "started_at": self.started_at,
            "ready_at": self.ready_at,
            "uptime_s": round(now - self.started_at, 1) if self.started_at else None,
            "health": dict(self.health),
            "host": self._host,
            "port": self._port,
            "log_path": str(self.log_path) if self.log_path else None,
        }

    # -- 启动 --------------------------------------------------------------
    async def start(self, argv: list[str], cwd: str, host: str, port: int) -> dict[str, Any]:
        if self.alive():
            raise RuntimeError("serve 已经在运行，请先停止")
        self.bind(host, port)
        log_handle: Any = subprocess.DEVNULL
        if self.log_dir is not None:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            stamp = time.strftime("%Y%m%d-%H%M%S")
            self.log_path = self.log_dir / f"serve-{stamp}.log"
            log_handle = open(self.log_path, "wb")  # noqa: SIM115 - 交给子进程持有
        proc = subprocess.Popen(
            argv,
            cwd=cwd or None,
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            creationflags=CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP,
        )
        if log_handle is not subprocess.DEVNULL:
            log_handle.close()  # 子进程已继承句柄，父进程不再需要
        with self._lock:
            self._proc = proc
        self.state = "starting"
        self.argv = list(argv)
        self.started_at = time.time()
        self.ready_at = None
        self.exit_code = None
        self.health = {"ok": False, "ts": 0.0}
        self.external_pid = None
        self._stopping = False
        self.logbus.append(f"启动 serve：{subprocess.list2cmdline(argv)}", "launcher")
        if self.log_path is not None:
            threading.Thread(target=self._tail_log, args=(self.log_path, proc), daemon=True,
                             name="serve-tail").start()
        threading.Thread(target=self._wait_exit, args=(proc,), daemon=True,
                         name="serve-wait").start()
        self._ensure_health_loop()
        return self.snapshot()

    def _tail_log(self, path: Path, proc: subprocess.Popen[bytes]) -> None:
        """尾随子进程日志文件，逐行交给 LogBus。"""

        try:
            handle = open(path, "r", encoding="utf-8", errors="replace", newline="")
        except OSError:
            return
        buffer = ""
        exited_at: float | None = None
        with handle:
            while True:
                chunk = handle.read(65536)
                if chunk:
                    buffer += chunk
                    while "\n" in buffer:
                        line, buffer = buffer.split("\n", 1)
                        self.logbus.append(line, "serve")
                    continue
                if proc.poll() is not None:
                    if exited_at is None:
                        exited_at = time.time()
                    elif time.time() - exited_at > 1.0:
                        break
                elif exited_at is None:
                    exited_at = None
                time.sleep(TAIL_POLL_SECONDS)
        # 收尾：写出残留的半个行。
        if buffer.strip():
            self.logbus.append(buffer, "serve")

    def _wait_exit(self, proc: subprocess.Popen[bytes]) -> None:
        code = proc.wait()
        with self._lock:
            if self._proc is proc:
                self.exit_code = code
                if self.state in ("starting", "ready"):
                    self.state = "stopped" if (code == 0 or self._stopping) else "crashed"
                self._proc = None
        self.logbus.append(f"serve 退出，退出码 {code}", "launcher")

    # -- 健康轮询 ----------------------------------------------------------
    def _ensure_health_loop(self) -> None:
        if self._health_task is None or self._health_task.done():
            self._health_task = asyncio.create_task(self._health_loop())

    async def _health_loop(self) -> None:
        while True:
            if self.state == "stopped" and not self.alive():
                return
            delay = STARTING_POLL_SECONDS if self.state == "starting" else READY_POLL_SECONDS
            outcome = await self._probe_health()
            self.health = outcome
            if outcome["ok"] and self.state == "starting":
                self.state = "ready"
                self.ready_at = time.time()
                elapsed = (self.ready_at or 0) - (self.started_at or self.ready_at or 0)
                self.logbus.append(f"serve 就绪，用时 {elapsed:.1f}s", "launcher")
            if self.state == "external" and not self.alive():
                if not port_in_use(self._host, self._port, timeout=0.3):
                    self.state = "stopped"
                    self.external_pid = None
                    self.logbus.append("外部 serve 已不在监听，退出只观测模式", "launcher")
                    return
            if not self.alive() and self.state != "external":
                if self.state in ("starting", "ready"):
                    await asyncio.sleep(0.2)
                if not self.alive():
                    return
            await asyncio.sleep(delay)

    async def _probe_health(self) -> dict[str, Any]:
        import httpx

        url = f"http://{self._host}:{self._port}/health"
        try:
            async with httpx.AsyncClient(timeout=1.5) as client:
                response = await client.get(url)
            data = response.json()
            ok = response.status_code == 200 and str(data.get("status", "")).lower() == "ok"
            return {"ok": ok, "status_code": response.status_code, "detail": data, "ts": time.time()}
        except Exception as exc:  # noqa: BLE001 - 探测失败是常态
            return {"ok": False, "error": str(exc), "ts": time.time()}

    # -- 外部实例 ----------------------------------------------------------
    async def check_external(self, host: str, port: int) -> None:
        """无子进程时检查端口：被占 → external（只观测），释放 → 回到 stopped。"""

        if self.alive():
            return
        self.bind(host, port)
        in_use = await asyncio.to_thread(port_in_use, host, port, 0.5)
        if in_use:
            if self.state == "external" and self.external_pid:
                return
            self.external_pid = await asyncio.to_thread(listening_pid, port)
            self.state = "external"
            self.started_at = None
            self.ready_at = None
            name = ""
            if self.external_pid:
                image = await asyncio.to_thread(process_image_name, self.external_pid)
                name = f"（{image}，PID {self.external_pid}）" if image else f"（PID {self.external_pid}）"
            self.logbus.append(
                f"检测到端口 {port} 已被占用{name}：进入只观测模式（无日志、无停止权）", "launcher"
            )
            self._ensure_health_loop()
            return
        if self.state == "external":
            self.state = "stopped"
            self.external_pid = None
            self.logbus.append("外部实例已释放端口，退出只观测模式", "launcher")

    # -- 停止 --------------------------------------------------------------
    async def stop(self) -> dict[str, Any]:
        proc = self._proc
        if proc is None:
            if self.state == "external":
                raise RuntimeError("serve 由外部启动，launcher 无法停止它")
            return self.snapshot()
        self._stopping = True
        self.logbus.append("正在停止 serve（terminate）", "launcher")
        try:
            proc.terminate()
            deadline = time.time() + KILL_GRACE_SECONDS
            while time.time() < deadline:
                if proc.poll() is not None:
                    break
                await asyncio.sleep(0.1)
            if proc.poll() is None:
                self.logbus.append("serve 未在期限内退出，强制结束（kill）", "launcher")
                proc.kill()
                await asyncio.sleep(0.2)
        except OSError:
            pass
        finally:
            # _wait_exit 也可能先到；这里兜底成「用户主动停止」而不是崩溃。
            with self._lock:
                if self._proc is proc and proc.poll() is not None:
                    self.exit_code = proc.returncode
                    self.state = "stopped"
                    self._proc = None
            self._stopping = False
        return self.snapshot()

    async def shutdown(self) -> None:
        """launcher 退出时只停探测任务；serve 子进程继续常驻。"""

        if self._health_task is not None:
            self._health_task.cancel()
            try:
                await self._health_task
            except asyncio.CancelledError:
                pass
            self._health_task = None