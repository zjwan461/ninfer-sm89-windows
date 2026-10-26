"""模型下载：从 Hugging Face（或其镜像）拉取 `.ninfer` 产物到本地模型目录。

要点：

- HTTP Range 断点续传，等价于仓库 ``scripts/download-*.bat`` 里的 ``curl -C -``；
- 只接受 ``owner/name`` 形式的仓库与裸文件名，拒绝路径穿越；
- 单进程内存作业表，最多两个并发；进度以字节计数 + EMA 速度暴露，前端轮询。
"""

from __future__ import annotations

import asyncio
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

CHUNK_BYTES = 1 << 20
MAX_CONCURRENT = 2

_REPO_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")
_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

TERMINAL = ("done", "failed", "cancelled")


@dataclass
class DownloadJob:
    id: str
    repo: str
    filename: str
    dest: Path
    url: str
    state: str = "queued"
    total: int | None = None
    done: int = 0
    error: str = ""
    proxy: str = ""
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    speed_bps: float = 0.0
    cancel: asyncio.Event = field(default_factory=asyncio.Event)
    #: 暂停与取消分离：暂停让读取循环自己收尾（文件句柄正常关闭），之后可原地续传。
    pause_requested: asyncio.Event = field(default_factory=asyncio.Event)
    _last_ts: float = 0.0
    _last_done: int = 0

    def sample(self) -> None:
        """EMA 速度：避免瞬时抖动让 UI 跳。"""

        now = time.time()
        if self._last_ts and now > self._last_ts:
            instant = (self.done - self._last_done) / (now - self._last_ts)
            self.speed_bps = instant if self.speed_bps <= 0 else self.speed_bps * 0.7 + instant * 0.3
        self._last_ts = now
        self._last_done = self.done

    def to_json(self) -> dict[str, Any]:
        remaining = None
        if self.total and self.speed_bps > 0 and self.done < self.total:
            remaining = (self.total - self.done) / self.speed_bps
        return {
            "id": self.id,
            "repo": self.repo,
            "filename": self.filename,
            "dest": str(self.dest),
            "url": self.url,
            "state": self.state,
            "total": self.total,
            "done": self.done,
            "percent": (self.done / self.total * 100) if self.total else None,
            "speed_bps": self.speed_bps,
            "eta_s": remaining,
            "error": self.error,
            "proxy": self.proxy,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }


class DownloadManager:
    def __init__(self) -> None:
        self._jobs: dict[str, DownloadJob] = {}
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._sem = asyncio.Semaphore(MAX_CONCURRENT)

    # -- 查询 --------------------------------------------------------------
    def snapshot(self) -> list[dict[str, Any]]:
        ordered = sorted(self._jobs.values(), key=lambda job: job.created_at)
        return [job.to_json() for job in ordered]

    def get(self, job_id: str) -> DownloadJob | None:
        return self._jobs.get(job_id)

    # -- 启动 --------------------------------------------------------------
    def start(
        self, repo: str, filename: str, dest_root: str, endpoint: str, proxy: str = ""
    ) -> dict[str, Any]:
        repo = (repo or "").strip().strip("/")
        filename = (filename or "").strip().strip("/")
        proxy = (proxy or "").strip()
        if not _REPO_PATTERN.match(repo):
            raise ValueError("仓库要写成 owner/name，例如 jgamboa/Qwen3.8-27B-NInfer-4090")
        if not _NAME_PATTERN.match(filename):
            raise ValueError("文件名只能是裸文件名（不能带路径分隔符或 ..）")
        if not endpoint.startswith(("http://", "https://")):
            raise ValueError("镜像地址必须以 http:// 或 https:// 开头")
        if proxy and not proxy.startswith(("http://", "https://")):
            raise ValueError("代理地址必须以 http:// 或 https:// 开头，例如 http://127.0.0.1:7897")

        root = Path(dest_root).expanduser()
        try:
            root.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise ValueError(f"无法创建保存目录：{exc}") from exc

        job = DownloadJob(
            id=uuid.uuid4().hex[:12],
            repo=repo,
            filename=filename,
            dest=root / filename,
            url=f"{endpoint.rstrip('/')}/{repo}/resolve/main/{filename}",
            proxy=proxy,
        )
        self._jobs[job.id] = job
        self._tasks[job.id] = asyncio.create_task(self._run(job))
        return job.to_json()

    async def _run(self, job: DownloadJob) -> None:
        async with self._sem:
            if job.cancel.is_set():
                job.state = "cancelled"
                job.finished_at = time.time()
                return
            if job.pause_requested.is_set():
                job.state = "paused"
                return
            job.state = "running"
            if job.started_at is None:
                job.started_at = time.time()
            try:
                finished = await self._download(job)
                if finished:
                    job.state = "done"
            except asyncio.CancelledError:
                if job.state != "paused":
                    job.state = "cancelled"
                raise
            except Exception as exc:  # noqa: BLE001 - 网络/磁盘错误都要显示给用户
                job.state = "failed"
                job.error = str(exc) or exc.__class__.__name__
            finally:
                if job.state in TERMINAL:
                    job.finished_at = time.time()

    async def _download(self, job: DownloadJob) -> bool:
        """返回是否完整下完；暂停/取消时返回 False（文件保留，供续传）。"""

        import httpx

        existing = job.dest.stat().st_size if job.dest.is_file() else 0
        headers = {"Range": f"bytes={existing}-"} if existing else {}
        timeout = httpx.Timeout(connect=20.0, read=120.0, write=60.0, pool=20.0)

        # proxy 为空时不显式设置：httpx 默认 trust_env=True，会沿用进程里的
        # HTTPS_PROXY/HTTP_PROXY（与 bootstrap 下载 Python 时的行为一致）。
        client_kwargs: dict[str, Any] = {"follow_redirects": True, "timeout": timeout}
        if job.proxy:
            client_kwargs["proxy"] = job.proxy

        async with httpx.AsyncClient(**client_kwargs) as client:
            async with client.stream("GET", job.url, headers=headers) as response:
                if response.status_code == 416:  # 已完整：服务器拒绝越界 Range
                    job.done = job.total = existing
                    return True
                response.raise_for_status()

                if response.status_code == 206 and existing:
                    mode = "ab"
                else:  # 200：服务器不支持续传，从头写
                    mode = "wb"
                    existing = 0
                length = response.headers.get("Content-Length")
                job.total = existing + int(length) if length and length.isdigit() else None
                job.done = existing
                job._last_ts = time.time()
                job._last_done = existing

                with job.dest.open(mode) as handle:
                    async for chunk in response.aiter_bytes(CHUNK_BYTES):
                        if job.cancel.is_set():
                            job.state = "cancelled"
                            return False
                        if job.pause_requested.is_set():
                            job.state = "paused"
                            return False
                        await asyncio.to_thread(handle.write, chunk)
                        job.done += len(chunk)
                        job.sample()

        if job.total is not None and job.done < job.total:
            raise RuntimeError(f"下载不完整：{job.done}/{job.total} 字节，可继续续传")
        return True

    # -- 控制 --------------------------------------------------------------
    def pause(self, job_id: str) -> bool:
        """请求暂停：读循环在下个分块处自行收尾，已下载字节保留用于续传。"""

        job = self._jobs.get(job_id)
        if job is None or job.state not in ("running", "queued"):
            return False
        job.pause_requested.set()
        return True

    async def resume(self, job_id: str) -> bool:
        """继续（暂停/失败/取消后都可用）：从磁盘上已有的字节处接着 Range 下载。

        先等旧的读取任务退出，避免两个任务同时写同一个文件。
        """

        job = self._jobs.get(job_id)
        if job is None or job.state not in ("paused", "failed", "cancelled"):
            return False

        previous = self._tasks.get(job_id)
        if previous is not None and not previous.done():
            try:
                await asyncio.wait_for(asyncio.shield(previous), timeout=5.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                pass
            except Exception:  # noqa: BLE001 - 旧任务的失败不影响续传
                pass

        job.pause_requested.clear()
        job.cancel.clear()
        job.error = ""
        job.finished_at = None
        job.state = "queued"
        self._tasks[job_id] = asyncio.create_task(self._run(job))
        return True

    def cancel(self, job_id: str) -> bool:
        job = self._jobs.get(job_id)
        if job is None or job.state in TERMINAL:
            return False
        job.cancel.set()
        task = self._tasks.get(job_id)
        if task is not None and not task.done():
            task.cancel()
        else:  # 暂停中的任务没有活跃 task，直接落终态
            job.state = "cancelled"
            job.finished_at = time.time()
        return True

    def clear(self, job_id: str) -> bool:
        job = self._jobs.get(job_id)
        if job is None or job.state not in TERMINAL:
            return False
        del self._jobs[job_id]
        self._tasks.pop(job_id, None)
        return True

    async def shutdown(self) -> None:
        for task in list(self._tasks.values()):
            if not task.done():
                task.cancel()
        for task in list(self._tasks.values()):
            try:
                await task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
        self._tasks.clear()