#!/usr/bin/env python3
# -- coding: utf-8 --

import sys
import os
import time
import json
import socket
import threading
import subprocess
from typing import Optional, List, Awaitable

# 可选：使用 psutil 更稳（不存在也没关系）
try:
    import psutil  # type: ignore
except Exception:
    psutil = None  # noqa

from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
import uvicorn

from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError
import asyncio
import urllib.request
import urllib.error
import logging

# -----------------------------
# 日志：提升到 DEBUG，统一格式
# -----------------------------
LOG_FMT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
logging.basicConfig(
    level=logging.DEBUG,
    format=LOG_FMT,
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,  # 覆盖任何已有配置
)
logger = logging.getLogger("http_run")

# === 基础路径 ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_PATH = os.path.join(BASE_DIR, "src")
sys.path.insert(0, SRC_PATH)
logger.debug(f"BASE_DIR={BASE_DIR}")
logger.debug(f"SRC_PATH={SRC_PATH}")

# ====== webui.json 路径（以 http_run.py 所在目录为基准）======
WEBUI_JSON_PATH = os.path.join(BASE_DIR, "src", "webui", "components", "webui.json")
logger.debug(f"WEBUI_JSON_PATH={WEBUI_JSON_PATH}")

# ====== webui.py 路径与解释器 ======
WEBUI_SCRIPT_PATH = os.environ.get("WEBUI_SCRIPT_PATH", os.path.join(BASE_DIR, "webui.py"))
PYTHON_EXE = os.environ.get("WEBUI_PYTHON", sys.executable)
logger.debug(f"WEBUI_SCRIPT_PATH={WEBUI_SCRIPT_PATH}")
logger.debug(f"PYTHON_EXE={PYTHON_EXE}")

# ====== WebUI host/port（默认值，可被请求覆盖）======
DEFAULT_WEBUI_HOST = os.environ.get("WEBUI_HOST", "127.0.0.1")
DEFAULT_WEBUI_PORT = int(os.environ.get("WEBUI_PORT", "7788"))  # 与 webui.py 默认一致
logger.debug(f"DEFAULT_WEBUI_HOST={DEFAULT_WEBUI_HOST}, DEFAULT_WEBUI_PORT={DEFAULT_WEBUI_PORT}")

# ====== 可选 pid 文件，便于管理由本脚本启动的 webui.py 实例 ======
PID_DIR = os.path.join(BASE_DIR, "tmp")
PID_FILE = os.path.join(PID_DIR, "webui.pid")
logger.debug(f"PID_FILE={PID_FILE}")

# -----------------------------
# 文件/端口/HTTP 工具
# -----------------------------
def ensure_dir_for(path: str):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)
        logger.debug(f"Created directory: {d}")

def write_or_update_webui_json(cdp_URP_value: Optional[str]):
    """
    将 cdp_URP 写入/更新到 WEBUI_JSON_PATH。
    若 cdp_URP_value 为 None，则保持文件原样（如果存在）。
    """
    logger.info("写入 webui.json ...")
    ensure_dir_for(WEBUI_JSON_PATH)
    data = {}
    if os.path.exists(WEBUI_JSON_PATH):
        try:
            with open(WEBUI_JSON_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            logger.debug(f"现有 webui.json 内容: {data}")
        except Exception as e:
            logger.warning(f"读取 webui.json 失败，将覆盖写入: {e}")
            data = {}
    if cdp_URP_value is not None:
        data["cdp_URP"] = cdp_URP_value
    with open(WEBUI_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info(f"webui.json 已写入: {data}")

def is_port_free(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.3)
        res = s.connect_ex((host, port)) != 0
        logger.debug(f"is_port_free({host}, {port}) -> {res}")
        return res

def find_free_port(host: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        free_port = s.getsockname()[1]
        logger.debug(f"find_free_port({host}) -> {free_port}")
        return free_port

def wait_http_ok(url: str, timeout_s: float = 30.0, interval_s: float = 0.5) -> bool:
    """
    轮询 GET 根路径是否可达；打印失败原因，便于排查。
    """
    logger.info(f"[probe] 开始探活：{url} (timeout={timeout_s}s)")
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2.0) as resp:
                logger.info(f"[probe] {url} -> HTTP {resp.status}")
                if 200 <= resp.status < 500:
                    return True
        except urllib.error.HTTPError as e:
            # 已有 HTTP 响应（例如 404）也算服务就绪
            logger.info(f"[probe] {url} -> HTTP {e.code} (OK)")
            return True
        except Exception as e:
            logger.debug(f"[probe] {url} not ready: {e.__class__.__name__}: {e}")
        time.sleep(interval_s)
    logger.warning(f"[probe] 超时：{url} 仍未就绪")
    return False

# -----------------------------
# 进程管理：查杀 webui.py
# -----------------------------
def _kill_pid(pid: int, force: bool = False):
    logger.debug(f"_kill_pid(pid={pid}, force={force})")
    if psutil:
        try:
            p = psutil.Process(pid)
            if force:
                p.kill()
            else:
                p.terminate()
            try:
                p.wait(timeout=3)
                logger.debug(f"PID {pid} 已结束")
            except Exception:
                if not force:
                    logger.debug(f"PID {pid} 超时未结束，执行 kill")
                    p.kill()
        except Exception as e:
            logger.debug(f"结束 PID {pid} 异常: {e}")
    else:
        try:
            if os.name == "nt":
                subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                import signal
                os.kill(pid, signal.SIGTERM)
                time.sleep(0.5)
                os.kill(pid, signal.SIGKILL)
            logger.debug(f"PID {pid} 已结束（非 psutil）")
        except Exception as e:
            logger.debug(f"结束 PID {pid} 异常（非 psutil）: {e}")

def _pids_listening_on_port(port: int) -> List[int]:
    logger.debug(f"_pids_listening_on_port({port})")
    pids = []
    if psutil:
        try:
            for c in psutil.net_connections(kind="inet"):
                if c.laddr and c.laddr.port == port and c.status == psutil.CONN_LISTEN and c.pid:
                    pids.append(c.pid)
        except Exception as e:
            logger.debug(f"net_connections 枚举异常: {e}")
    else:
        # Windows: netstat -ano | findstr :<port>
        try:
            if os.name == "nt":
                out = subprocess.check_output(["netstat", "-ano"], text=True, creationflags=subprocess.CREATE_NO_WINDOW)
                for line in out.splitlines():
                    if f":{port} " in line and "LISTENING" in line:
                        parts = line.split()
                        pid = int(parts[-1])
                        pids.append(pid)
        except Exception as e:
            logger.debug(f"netstat 枚举异常: {e}")
    pids = list(sorted(set(pids)))
    logger.debug(f"监听 {port} 的 PID 列表：{pids}")
    return pids

def kill_existing_webui(host: str, port: int):
    """
    优先：pid 文件 → 命令行包含 webui.py → 监听端口的进程
    """
    logger.info("🛑 尝试停止已有 webui.py ...")
    # 1) pid 文件
    if os.path.exists(PID_FILE):
        try:
            with open(PID_FILE, "r", encoding="utf-8") as f:
                pid = int(f.read().strip())
            logger.debug(f"从 PID_FILE 读取 PID={pid}")
            _kill_pid(pid)
        except Exception as e:
            logger.debug(f"读取/结束 PID_FILE 失败: {e}")
        try:
            os.remove(PID_FILE)
            logger.debug(f"已删除 PID_FILE: {PID_FILE}")
        except Exception:
            pass

    # 2) 命令行包含 webui.py 的进程
    if psutil:
        try:
            for p in psutil.process_iter(attrs=["pid", "name", "cmdline"]):
                try:
                    cmdline = " ".join(p.info.get("cmdline") or [])
                    if "webui.py" in cmdline:
                        logger.debug(f"结束命令行包含 webui.py 的 PID={p.info['pid']}")
                        _kill_pid(p.info["pid"])
                except Exception:
                    continue
        except Exception as e:
            logger.debug(f"遍历进程异常: {e}")

    # 3) 按端口杀
    for pid in _pids_listening_on_port(port):
        logger.debug(f"结束监听 {port} 的 PID={pid}")
        _kill_pid(pid, force=True)

    # 等端口释放
    for _ in range(20):  # 最多等 4 秒
        if is_port_free(host, port):
            logger.debug(f"端口 {port} 已释放")
            break
        time.sleep(0.2)

# -----------------------------
# 子进程日志转发（避免 PIPE 卡住 + 方便排查）
# -----------------------------
def _pump_output(proc: subprocess.Popen, prefix: str):
    def _reader(stream):
        for line in iter(stream.readline, b""):
            try:
                sys.stdout.write(f"{prefix}{line.decode(errors='ignore')}")
                sys.stdout.flush()
            except Exception:
                pass
    if proc.stdout:
        t = threading.Thread(target=_reader, args=(proc.stdout,), daemon=True)
        t.start()

def _watch_exit(proc: subprocess.Popen, label: str):
    def _waiter():
        code = proc.wait()
        logger.warning(f"{label} 退出，returncode={code}")
    threading.Thread(target=_waiter, daemon=True).start()

# -----------------------------
# 拉起 webui.py（用 argparse 的 --ip/--port）
# -----------------------------
def launch_webui_process(host: str, port: int) -> subprocess.Popen:
    chosen_port = port
    if not is_port_free(host, chosen_port):
        logger.warning(f"端口 {chosen_port} 被占用，自动选择空闲端口")
        chosen_port = find_free_port(host)

    ensure_dir_for(PID_FILE)

    # 环境变量：强制无缓冲，确保日志实时输出
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    # 注意：不要 DETACHED_PROCESS，否则没有控制台输出也不易调试
    creationflags = 0

    cmd = [PYTHON_EXE, WEBUI_SCRIPT_PATH, "--ip", host, "--port", str(chosen_port)]
    logger.info("🚀 启动 webui.py：%s", " ".join(cmd))
    logger.debug(f"cwd={BASE_DIR}")
    proc = subprocess.Popen(
        cmd,
        cwd=BASE_DIR,
        stdout=subprocess.PIPE,         # ← 用线程实时消费
        stderr=subprocess.STDOUT,
        env=env,
        creationflags=creationflags
    )
    _pump_output(proc, "[webui.py] ")
    _watch_exit(proc, "webui.py")

    # 记录 pid
    try:
        with open(PID_FILE, "w", encoding="utf-8") as f:
            f.write(str(proc.pid))
        logger.debug(f"写入 PID_FILE={PID_FILE}, pid={proc.pid}")
    except Exception as e:
        logger.debug(f"写 PID_FILE 失败: {e}")

    # 探活（根地址）
    root_url = f"http://{host}:{chosen_port}/"
    ok = wait_http_ok(root_url, timeout_s=30.0, interval_s=0.5)
    if not ok:
        logger.warning("webui.py 探活未成功（端口可能稍后才就绪），稍后 Playwright 还会再试。")
    else:
        logger.info("webui.py 探活成功：%s", root_url)
    return proc, chosen_port

# -----------------------------
# Playwright 任务（参数化）
# -----------------------------
async def control_ui(
    task_text: str,
    webui_url: str,
    run_tab_selectors: Optional[List[str]] = None,
    submit_button_text: str = "Submit Task",
    open_browser_headless: bool = False,
    max_wait_ms: int = 10_000,
    settle_delay_ms: int = 3_000,
):
    """
    通过 Playwright 操作 WebUI：
    - 打开 WebUI
    - 切到“Run Agent”标签（用多个 Selector 兜底）
    - 填写第一个 textarea
    - 点击 “Submit Task” 按钮
    """
    logger.info(f"Playwright 将连接：{webui_url}")
    run_tab_selectors = run_tab_selectors or [
        "#component-82-button",
        "button:has-text('Run Agent')",
        "[id*='component'][id$='-button']:has-text('Run Agent')",
    ]

    logger.debug("等待 WebUI 就绪（Playwright）...")
    time.sleep(1.0)

    async with async_playwright() as p:
        logger.debug("启动 Chromium（headless=%s）", open_browser_headless)
        browser = await p.chromium.launch(headless=open_browser_headless)
        context = await browser.new_context()
        page = await context.new_page()

        logger.info(f"🌐 打开 WebUI 页面: {webui_url}")
        await page.goto(webui_url, wait_until="domcontentloaded")

        if settle_delay_ms > 0:
            logger.debug(f"点击 Run Agent 前等待 {settle_delay_ms} ms")
            await asyncio.sleep(settle_delay_ms / 1000.0)

        last_err = None
        for sel in run_tab_selectors:
            try:
                logger.debug(f"尝试点击标签: {sel}")
                await page.click(sel, timeout=max_wait_ms)
                last_err = None
                logger.debug("点击成功")
                break
            except (TimeoutError, PlaywrightTimeoutError) as e:
                logger.debug(f"点击失败（{sel}）：{e}")
                last_err = e
            except Exception as e:
                logger.debug(f"点击异常（{sel}）：{e}")
                last_err = e

        if last_err:
            await browser.close()
            raise RuntimeError(
                f"未能点击到 'Run Agent' 标签，请检查 selector 列表: {run_tab_selectors}. err={last_err}"
            )

        logger.debug("等待 textarea 出现 ...")
        await page.wait_for_selector("textarea", timeout=max_wait_ms)

        logger.debug("填写任务内容 ...")
        textareas = await page.query_selector_all("textarea")
        if not textareas:
            await browser.close()
            raise RuntimeError("页面中没有找到任何 textarea。")
        await textareas[0].fill(task_text)

        logger.debug("点击提交按钮 ...")
        try:
            await page.click(f"button:has-text('{submit_button_text}')", timeout=max_wait_ms)
        except Exception as e:
            logger.debug(f"按文案点击失败：{e}，尝试点击第一个按钮")
            buttons = await page.query_selector_all("button")
            if not buttons:
                await browser.close()
                raise RuntimeError("页面中没有找到任何 button（无法点击提交）。")
            await buttons[0].click()

        logger.info("✅ 任务已提交，等待 5 秒观察 ...")
        await page.wait_for_timeout(5_000)
        await browser.close()

# -----------------------------
# 后台事件循环线程（用于跑异步任务）
# -----------------------------
class AsyncWorker:
    """在独立线程中运行一个 asyncio 事件循环，并允许从主线程提交协程。"""

    def __init__(self):
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        logger.debug("AsyncWorker 事件循环线程已启动")

    def _run_loop(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        logger.debug("AsyncWorker loop running forever ...")
        self.loop.run_forever()

    def submit_coro(self, coro: Awaitable):
        """线程安全地提交协程，返回 concurrent.futures.Future"""
        if self.loop is None:
            raise RuntimeError("AsyncWorker 的事件循环尚未就绪。")
        logger.debug("提交协程到后台事件循环 ...")
        return asyncio.run_coroutine_threadsafe(coro, self.loop)

# -----------------------------
# FastAPI 定义
# -----------------------------
app = FastAPI(title="UI Task Runner", version="1.8.0")
worker = AsyncWorker()
_task_lock = asyncio.Lock()
_proc_lock = threading.Lock()

class StartTaskReq(BaseModel):
    task: str
    # 写入 webui.json 的字段
    cdp_URP: Optional[str] = None
    # WebUI 参数（可选）
    host: Optional[str] = None
    port: Optional[int] = None
    # Playwright 参数
    headless: Optional[bool] = False
    run_tab_selectors: Optional[List[str]] = None
    submit_button_text: Optional[str] = "Submit Task"
    max_wait_ms: Optional[int] = 10_000
    settle_delay_ms: Optional[int] = 3_000

class StartTaskResp(BaseModel):
    status: str
    message: str
    webui_url: Optional[str] = None
    pid: Optional[int] = None

@app.get("/health")
def health():
    return {"status": "ok", "script": WEBUI_SCRIPT_PATH, "json": WEBUI_JSON_PATH}

@app.get("/version")
def version():
    return {"version": app.version}

@app.post("/start", response_model=StartTaskResp)
def start_task(req: StartTaskReq):
    logger.info("收到 /start 请求")
    logger.debug(f"请求体：{req.json()}")
    if not req.task or not req.task.strip():
        raise HTTPException(status_code=400, detail="task 不能为空")

    host = (req.host or DEFAULT_WEBUI_HOST).strip()
    try:
        port = int(req.port) if req.port else DEFAULT_WEBUI_PORT
    except Exception:
        port = DEFAULT_WEBUI_PORT
    logger.info(f"目标 WebUI: {host}:{port}")

    # ① 停止已有 webui.py
    try:
        with _proc_lock:
            kill_existing_webui(host, port)
    except Exception as e:
        logger.exception("停止 webui.py 失败")
        raise HTTPException(status_code=500, detail=f"停止 webui.py 失败：{e}")

    # ② 写 webui.json
    try:
        write_or_update_webui_json(req.cdp_URP)
    except Exception as e:
        logger.exception("写入 webui.json 失败")
        raise HTTPException(status_code=500, detail=f"写入 webui.json 失败：{e}")

    # ③ 重新唤起 webui.py（传入 --ip/--port）
    try:
        with _proc_lock:
            proc, real_port = launch_webui_process(host, port)
            webui_url = f"http://{host}:{real_port}"
    except Exception as e:
        logger.exception("启动 webui.py 失败")
        raise HTTPException(status_code=500, detail=f"启动 webui.py 失败：{e}")

    # ④ Playwright 控制（异步执行）
    async def _job():
        async with _task_lock:
            await control_ui(
                task_text=req.task.strip(),
                webui_url=webui_url,
                run_tab_selectors=req.run_tab_selectors,
                submit_button_text=req.submit_button_text or "Submit Task",
                open_browser_headless=bool(req.headless),
                max_wait_ms=int(req.max_wait_ms or 10_000),
                settle_delay_ms=int(req.settle_delay_ms or 3_000),
            )

    try:
        worker.submit_coro(_job())
        msg = f"任务已提交并在后台执行（WebUI: {webui_url}）。"
        if req.cdp_URP is not None:
            msg += f" 已更新 {WEBUI_JSON_PATH} 的 cdp_URP。"
        logger.info(msg)
        return StartTaskResp(status="accepted", message=msg, webui_url=webui_url, pid=proc.pid)
    except Exception as e:
        logger.exception("提交任务失败")
        raise HTTPException(status_code=500, detail=f"提交任务失败：{e}")

# -----------------------------
# 入口：仅启动 API（不启动 WebUI）
# -----------------------------
def start_api_server(host: str = None, port: int = None):
    api_host = host or os.environ.get("API_HOST", "0.0.0.0")
    api_port = int(port or os.environ.get("API_PORT", "9000"))
    logger.info(f"🛰️ 启动 API 服务：http://{api_host}:{api_port}")
    uvicorn.run(
        app,
        host=api_host,
        port=api_port,
        log_level="debug",  # 让 uvicorn 走 DEBUG
    )

if __name__ == "__main__":
    # 这里只启动 API，按请求去：杀 webui.py → 写 JSON → 启 webui.py（--ip/--port）→ Playwright 控制
    start_api_server()
