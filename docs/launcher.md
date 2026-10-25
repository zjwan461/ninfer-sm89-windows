# 启动器（WebUI）

`start-launcher.bat` 启动一个只在 `127.0.0.1` 上监听的本地 WebUI，用来配置参数、启动/停止
`ninfer-serve.exe`、看日志与运行指标。它**不改动引擎**：只是构造命令行、监督子进程并代理观测
接口，所有推理仍由 NInfer 的 C++ 二进制完成。

## 快速开始

```bat
start-launcher.bat                  :: 默认 8100 端口，自动打开浏览器
start-launcher.bat --port 8101      :: 换 UI 端口
start-launcher.bat --no-browser     :: 不自动开浏览器
```

打开界面后：选一个档案（首次运行会自动导入四个内置预设）→ 确认模型文件 → 点「启动」。

## Python 从哪来

启动器需要 Python 3.10+（依赖 fastapi / uvicorn / httpx）。`tools/launcher/bootstrap.ps1`
按下面的顺序找一个可用的解释器：

1. 环境变量 `NINFER_LAUNCHER_PYTHON` 指定的解释器；
2. 之前自动下载的绿色版（`%LOCALAPPDATA%\NInfer-Launcher\runtime\python`）；
3. Windows 的 `py` 启动器（`py -3.13 / -3.12 / -3.11 / -3.10`）；
4. PATH 上的 `python` / `python3`；
5. 常见安装位置（python.org 安装器、miniconda/anaconda、miniforge、scoop）；
6. PATH 上的 `conda`（`conda info --base` 与其 env 列表）。

**都找不到时自动下载 python.org 官方绿色版**（`python-3.11.9-embed-amd64.zip`，约 11 MB），
解压到 launcher 私有目录、打开 `site-packages`、用 `get-pip.py` 装 pip，再把依赖装进这个私有环境。
系统 Python 不会被改动；整个过程只发生一次，之后直接复用。

| 环境变量 | 作用 |
|---|---|
| `NINFER_LAUNCHER_PYTHON` | 强制指定解释器路径 |
| `NINFER_LAUNCHER_HOME` | launcher 数据目录（默认 `%LOCALAPPDATA%\NInfer-Launcher`） |
| `NINFER_LAUNCHER_PYTHON_VERSION` | 绿色版版本（默认 `3.11.9`） |
| `NINFER_LAUNCHER_PYTHON_URL` | 直接指定 embeddable zip 的 URL |
| `HTTPS_PROXY` / `HTTP_PROXY` | 下载 Python 与 pip 依赖时使用的代理 |

下载慢时先设代理再启动，例如：

```bat
set HTTPS_PROXY=http://127.0.0.1:7897
start-launcher.bat
```

## 界面

| 标签 | 内容 |
|---|---|
| 总览 | 状态卡片（已停止/加载中/运行中/已崩溃/外部实例）、当前命令行、显存占用、`/v1` `/monitor` `/metrics` `/health` 链接、prefill/decode 速率、KV 页占用、草稿接受率、槽位与最近请求 |
| 配置 | 按组生成的参数表单（常规项直出、高级项折叠）、档案管理、命令行预览、校验结果 |
| 日志 | serve 实时日志（SSE）+ 启动器事件，关键词过滤、暂停、下载 |
| 测试 | 向运行中的 serve 发一次聊天请求，流式显示思考与回答，并给出 `timings`（prefill/decode tok/s、缓存复用、草稿接受） |

启动前会显示**完整命令行预览**；未填写的参数不会出现在命令行里，交给 serve 用自己的缺省值。

## 参数分组

| 组 | 关键参数 |
|---|---|
| 模型 | `--model-id`、`--vision`、`--vision-max-tokens`、`--chat-template` |
| 容量 | `--max-context`、`--kv-capacity`（auto/整数）、`--kv-dtype`、`--max-concurrency`（1–8）、`--max-pending-requests`、`--pending-timeout-ms` |
| 性能与投机解码 | `--prefill-chunk`、`--spec mtp/dflash/dflash2`、`--draft-tokens`、`--lm-head-draft`、`--ngram chain`（及子参数）、`--no-cuda-graph` |
| KV 与会话缓存 | `--device-state-slots`、`--host-state-slots`、`--host-kv-mib`、`--no-prefix-reuse`、长锚点与 Responses 存储上限（高级） |
| 请求默认值 | `--default-max-tokens`、`--default-thinking-budget`、`--preserve-thinking`、`--no-thinking`、采样器覆盖（高级） |
| 网络、日志与媒体 | `--host`、`--port`、`--api-key`、`--cors`、`--log-level`、请求日志、媒体缓存上限（高级） |

参数名与合法值以 `ninfer-serve.exe --help` 为准；已退役的 `--turn-checkpoints` 不出现在界面里。

界面会做这些校验（阻止启动）：并发必须是 1–8；选了 `--spec` 必须填 `--draft-tokens`（DFlash2 为
1–15）；`--auto-save-evicted` 需要 `--slot-save-path`；`--default-thinking-budget` 不能与
`--no-thinking` 同用。以下只提示不阻止：`--max-context` 超过 262144、`--draft-tokens` 没配
`--spec`、n-gram 子参数没开 `chain`、`--vision-max-tokens` 没开 `--vision`、`--greedy` 与
temperature 同时设置。

## 档案（profile）

- 档案 = 一整套参数（含引擎路径与模型文件），保存在
  `%LOCALAPPDATA%\NInfer-Launcher\profiles.json`；
- 首次运行自动导入四个内置预设，对应仓库的四个 `start-*.bat`：
  `Qwen3.8-27B int8 · 4080s/4090`、`Bonsai2-27B VL · 4080s/4090`；
- 支持另存为、删除、恢复内置预设；「导出 .bat」会生成与 `start-*.bat` 同形的脚本，
  可继续用 `NINFER_SERVER` / `%*` 覆盖。

## 进程生命周期

```
STOPPED ──启动──▶ STARTING ──/health 通过──▶ READY ──停止──▶ STOPPED
                      └──进程退出非 0──▶ CRASHED
端口已被外部进程占用 ──▶ EXTERNAL（只观测：可看健康与监控，无日志、无停止权）
```

- serve 的 stdout/stderr 直接重定向到 `%LOCALAPPDATA%\NInfer-Launcher\logs\serve-<时间戳>.log`，
  界面通过 SSE 实时展示；launcher 意外退出时 serve 不会因断管报错，日志也已落盘。
- 就绪判定：每秒轮询 `GET /health`；20 GB 的 artifact 首次加载通常需要几十秒，期间显示「加载中」。
- 停止：先 `terminate`，10 秒未退出再 `kill`；手动停止不会被误记为「崩溃」。
- 若端口被别的 `ninfer-serve.exe` 占用（例如手动跑的 `.bat`），页面会提示占用 PID，
  可以只观测、改端口，或在确认进程名后一键结束它。
- launcher 退出不会杀掉 serve：关掉界面后服务继续提供 API，下次启动自动进入只观测模式。

## 数据与状态

| 路径 | 内容 |
|---|---|
| `%LOCALAPPDATA%\NInfer-Launcher\profiles.json` | 参数档案与上次使用的档案 |
| `%LOCALAPPDATA%\NInfer-Launcher\settings.json` | UI 端口、引擎路径、模型目录、上次启动参数 |
| `%LOCALAPPDATA%\NInfer-Launcher\logs\serve-*.log` | 每次启动的 serve 原始日志 |
| `%LOCALAPPDATA%\NInfer-Launcher\runtime\python` | 自动下载的绿色版 Python 与依赖 |

## 验证与排查

```bash
# launcher 自身的单元测试（参数构造、校验、档案读写、进程状态机、日志总线）
python -m pytest tests/launcher -q
```

- 界面打不开：确认 8100 未被占用；重复启动只会打开已有实例的页面。
- 依赖装不上：设置 `HTTPS_PROXY` 后重试，或手动
  `"%LOCALAPPDATA%\NInfer-Launcher\runtime\python\python.exe" -m pip install -r tools/launcher/requirements.txt`。
- 想看 serve 到底收到什么命令行：总览页「当前命令行」或配置页「命令行预览」，与
  `logs\serve-*.log` 首行一致。