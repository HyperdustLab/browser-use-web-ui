#!/usr/bin/env python3
# -- coding: utf-8 --

import sys
import os
import time
import json
import socket
import threading
import subprocess
from typing import Optional, List, Awaitable, Dict, Any
from collections import deque
from dataclasses import dataclass, field

# Optional: Using psutil is more stable (it's okay if not present)
try:
    import psutil  # type: ignore
except Exception:
    psutil = None

from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
import uvicorn

from playwright.async_api import async_playwright
import asyncio
import urllib.request
import urllib.error
import logging
import uuid

# -----------------------------
# Logging
# -----------------------------
LOG_FMT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
logging.basicConfig(
    level=logging.DEBUG,
    format=LOG_FMT,
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
logger = logging.getLogger("http_run")

# === Base paths ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_PATH = os.path.join(BASE_DIR, "src")
sys.path.insert(0, SRC_PATH)

# ====== Path to webui.json (relative to http_run.py) ======
WEBUI_JSON_PATH = os.path.join(BASE_DIR, "src", "webui", "components", "webui.json")

# ====== Path to webui.py and interpreter ======
WEBUI_SCRIPT_PATH = os.environ.get("WEBUI_SCRIPT_PATH", os.path.join(BASE_DIR, "webui.py"))
PYTHON_EXE = os.environ.get("WEBUI_PYTHON", sys.executable)

# ====== WebUI host/port (default values, can be overridden by requests) ======
DEFAULT_WEBUI_HOST = os.environ.get("WEBUI_HOST", "127.0.0.1")
DEFAULT_WEBUI_PORT = int(os.environ.get("WEBUI_PORT", "7788"))  # Keep in sync with webui.py

# ====== Global Playwright state (shared between sessions) ======
_playwright = None  # type: ignore

# ====== Resume / Pause timing constants ======
ENTER_PROMPT_SUBSTR = "Press [Enter] to resume"

RESUME_WAIT_FOR_PROMPT_SECONDS = 2.0
RESUME_FALLBACK_SEND_ENTER = True
RESUME_OBSERVE_AFTER_SECONDS = 1.0
RESUME_GRACE_FOR_CONSUME_SECONDS = 0.3

# -----------------------------
# Utility functions
# -----------------------------
def ensure_dir_for(path: str):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def write_or_update_webui_json(
    cdp_URP_value: Optional[str],
    llm_model_name: Optional[str] = None,
    llm_base_url: Optional[str] = None,
):
    """
    Write/update cdp_URP, llm_model_name, llm_base_url to WEBUI_JSON_PATH.
    If parameter is None: keep original value; if empty string "": explicitly write empty string.
    """
    logger.info("Writing to webui.json.")
    ensure_dir_for(WEBUI_JSON_PATH)
    data: Dict[str, Any] = {}
    if os.path.exists(WEBUI_JSON_PATH):
        try:
            with open(WEBUI_JSON_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            logger.debug(f"Existing webui.json content: {data}")
        except Exception as e:
            logger.warning(f"Failed to read webui.json, will overwrite: {e}")
            data = {}
    if cdp_URP_value is not None:
        data["cdp_URP"] = cdp_URP_value
    if llm_model_name is not None:
        data["llm_model_name"] = llm_model_name
    if llm_base_url is not None:
        data["llm_base_url"] = llm_base_url
    with open(WEBUI_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info(f"webui.json written: {data}")

def is_port_free(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.3)
        return s.connect_ex((host, port)) != 0

def find_free_port(host: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return s.getsockname()[1]

def wait_http_ok(url: str, timeout_s: float = 60.0, interval_s: float = 1.0) -> bool:
    logger.info(f"[probe] Starting health check: {url} (timeout={timeout_s}s)")
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2.0) as resp:
                logger.info(f"[probe] {url} -> HTTP {resp.status}")
                if 200 <= resp.status < 500:
                    return True
        except urllib.error.HTTPError as e:
            logger.info(f"[probe] {url} -> HTTP {e.code} (OK)")
            return True
        except Exception:
            pass
        time.sleep(interval_s)
    logger.warning(f"[probe] Timeout: {url} not ready")
    return False

# -----------------------------
# Process helpers (per pid)
# -----------------------------
def _kill_pid(pid: int, force: bool = False):
    if psutil:
        try:
            p = psutil.Process(pid)
            (p.kill() if force else p.terminate())
            try:
                p.wait(timeout=3)
            except Exception:
                p.kill()
        except Exception:
            pass
    else:
        try:
            if os.name == "nt":
                subprocess.run(
                    ["taskkill", "/PID", str(pid), "/T", "/F"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            else:
                import signal
                os.kill(pid, signal.SIGTERM)
                time.sleep(0.5)
                os.kill(pid, signal.SIGKILL)
        except Exception:
            pass

# -----------------------------
# Session State (multi-task support)
# -----------------------------
@dataclass
class SessionState:
    session_id: str
    host: str
    port: int                    # WebUI HTTP 端口
    ws_port: Optional[int] = None  # WebUI WebSocket 端口（新增）

    webui_proc: Optional[subprocess.Popen] = None

    # Playwright browser state
    browser: Any = None
    context: Any = None
    page: Any = None

    browser_name: Optional[str] = None
    browser_channel: Optional[str] = None
    exec_path: Optional[str] = None
    cdp_url: Optional[str] = None

    # CLI / log based resume-prompt state
    enter_prompt_substr: str = ENTER_PROMPT_SUBSTR
    enter_prompt_event: threading.Event = field(default_factory=threading.Event)
    enter_prompt_lock: threading.Lock = field(default_factory=threading.Lock)
    enter_prompt_last_ts: float = 0.0

    last_pause_ts: float = 0.0
    enter_consumed_event: threading.Event = field(default_factory=threading.Event)
    enter_consumed_last_ts: float = 0.0

    last_lines: deque = field(default_factory=lambda: deque(maxlen=200))

_sessions: Dict[str, SessionState] = {}
_sessions_lock = threading.Lock()

def get_session(session_id: str) -> SessionState:
    with _sessions_lock:
        s = _sessions.get(session_id)
        if not s:
            raise KeyError(f"Session not found: {session_id}")
        return s

def create_session(session_id: str, host: str, port: int) -> SessionState:
    with _sessions_lock:
        if session_id in _sessions:
            raise ValueError(f"Session already exists: {session_id}")
        s = SessionState(session_id=session_id, host=host, port=port)
        _sessions[session_id] = s
        return s

def remove_session(session_id: str):
    with _sessions_lock:
        _sessions.pop(session_id, None)

# -----------------------------
# Subprocess output handling (per session)
# -----------------------------
def _pump_output_for_session(proc: subprocess.Popen, session: SessionState, prefix: str):
    def _reader(stream):
        for line in iter(stream.readline, b""):
            try:
                text = line.decode(errors="ignore")
            except Exception:
                text = ""

            # Print to stdout
            try:
                sys.stdout.write(f"{prefix}{text}")
                sys.stdout.flush()
            except Exception:
                pass

            if text:
                session.last_lines.append(text.rstrip("\n"))

            # 1) Detect CLI prompt "Press [Enter] to resume"
            if session.enter_prompt_substr in text:
                with session.enter_prompt_lock:
                    session.enter_prompt_last_ts = time.time()
                    session.enter_prompt_event.set()
                logger.debug(f"[{session.session_id}] Detected CLI resume prompt; event set.")

            # 2) Detect pause traces
            if ("Pause button clicked." in text) or ("Got Ctrl+C, paused" in text):
                session.last_pause_ts = time.time()

            # 3) Detect "Got Enter, resuming"
            if "Got Enter, resuming" in text:
                session.enter_consumed_last_ts = time.time()
                session.enter_consumed_event.set()
                logger.debug(f"[{session.session_id}] Detected ENTER already consumed by CLI.")

    if proc.stdout:
        t = threading.Thread(target=_reader, args=(proc.stdout,), daemon=True)
        t.start()

def _watch_exit(proc: subprocess.Popen, label: str):
    def _waiter():
        code = proc.wait()
        logger.warning(f"{label} exited with returncode={code}")
    threading.Thread(target=_waiter, daemon=True).start()

# -----------------------------
# Send "Enter" to a session's webui.py process
# -----------------------------
def send_enter_to_webui(session: SessionState) -> bool:
    """Write a newline to this session's webui.py stdin to simulate pressing Enter in CLI."""
    proc = session.webui_proc
    if proc is None:
        logger.warning(f"[{session.session_id}] send_enter_to_webui: webui_proc is None.")
        return False
    try:
        if proc.stdin:
            proc.stdin.write(b"\n")
            proc.stdin.flush()
            logger.info(f"[{session.session_id}] ▶️  Sent ENTER to webui.py stdin.")
            return True
        else:
            logger.warning(f"[{session.session_id}] send_enter_to_webui: stdin is None.")
            return False
    except Exception as e:
        logger.exception(f"[{session.session_id}] send_enter_to_webui failed: {e}")
        return False

# -----------------------------
# Launch webui.py for a session（包含独立 WS 端口）
# -----------------------------
def launch_webui_process_for_session(session: SessionState):
    host = session.host
    port = session.port

    # HTTP 端口
    chosen_port = port
    if not is_port_free(host, chosen_port):
        logger.warning(f"[{session.session_id}] Port {chosen_port} is occupied, automatically selecting a free port")
        chosen_port = find_free_port(host)

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    # WebSocket 端口（为该 session 单独占一个）
    ws_port = find_free_port(host)
    env["WEBUI_WS_PORT"] = str(ws_port)
    session.ws_port = ws_port

    # 从 webui.json 中读取 llm_base_url → 注入 OLLAMA_HOST
    try:
        if os.path.exists(WEBUI_JSON_PATH):
            with open(WEBUI_JSON_PATH, "r", encoding="utf-8") as f:
                _cfg = json.load(f)
            _ollama = (_cfg.get("llm_base_url") or "").strip()
            if _ollama.endswith("/"):
                _ollama = _ollama[:-1]
            for _sfx in ("/v1", "/v1/"):
                if _ollama.endswith(_sfx):
                    _ollama = _ollama[: -len(_sfx)]
            if _ollama:
                env["OLLAMA_HOST"] = _ollama
                logger.info(f"[{session.session_id}] Set OLLAMA_HOST={env['OLLAMA_HOST']} for subprocess")
        else:
            logger.info(f"[{session.session_id}] webui.json not found ({WEBUI_JSON_PATH}), skipping OLLAMA_HOST injection")
    except Exception as e:
        logger.warning(f"[{session.session_id}] Failed to read webui.json to set OLLAMA_HOST: {e}")

    cmd = [PYTHON_EXE, WEBUI_SCRIPT_PATH, "--ip", host, "--port", str(chosen_port)]
    logger.info(f"[{session.session_id}] 🚀 Launching webui.py: %s", " ".join(cmd))
    proc = subprocess.Popen(
        cmd,
        cwd=BASE_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        stdin=subprocess.PIPE,
        env=env,
        creationflags=0,
    )

    session.webui_proc = proc
    session.port = chosen_port

    _pump_output_for_session(proc, session, f"[webui.py:{session.session_id}] ")
    _watch_exit(proc, f"webui.py({session.session_id})")

    root_url = f"http://{host}:{chosen_port}/"
    ok = wait_http_ok(root_url, timeout_s=60.0, interval_s=1.0)
    if ok:
        logger.info(f"[{session.session_id}] webui.py health check successful: {root_url}")
    else:
        logger.warning(f"[{session.session_id}] webui.py health check not successful (may become ready later)")

    return proc, chosen_port

# -----------------------------
# Playwright: Launch/connect browser & submit task (per session)
# -----------------------------
async def _open_browser_and_submit_for_session(
    session: SessionState,
    webui_url: str,
    task_text: str,
    run_tab_selectors: Optional[List[str]] = None,
    headless: bool = False,
    max_wait_ms: int = 15_000,
    settle_delay_ms: int = 2_000,
    browser_name: str = "chromium",
    browser_channel: Optional[str] = None,
    executable_path: Optional[str] = None,
    browser_cdp_url: Optional[str] = None,
):
    """
    打开 / 连接浏览器 → 打开 WebUI → 点击 🤖 Run Agent → 找到任务输入框 → 填 task → 点击 ▶️ Submit Task
    - 支持 browser_cdp_url（外部浏览器）
    - 沿用你第一版 http_run.py 的 Run Agent 点击逻辑 + “检测不到任务输入框就持续点 Run Agent”
    """
    global _playwright

    TASK_PLACEHOLDER = "Enter your task here or provide assistance when asked."

    # ---------- 1. 准备 Playwright ----------
    if _playwright is None:
        _playwright = await async_playwright().start()

    # 如果这个 session 已经有浏览器，先关掉，保证一个任务一个干净环境
    if session.browser is not None:
        try:
            logger.info(f"[{session.session_id}] Closing existing browser before new task...")
            await session.browser.close()
        except Exception as e:
            logger.warning(f"[{session.session_id}] Error closing old browser: {e}")
        finally:
            session.browser = session.context = session.page = None

    # ---------- 2. 根据 browser_cdp_url 决定：连接外部浏览器 or 在容器内启动 ----------
    if browser_cdp_url:
        # 外部浏览器（CDP），比如 Docker 里连宿主机 Chrome
        logger.info(f"[{session.session_id}] Connecting external browser via CDP: {browser_cdp_url}")
        browser = await _playwright.chromium.connect_over_cdp(browser_cdp_url)

        if browser.contexts:
            context = browser.contexts[0]
        else:
            context = await browser.new_context()

        page = await context.new_page()

        session.browser = browser
        session.context = context
        session.page = page

        session.browser_name = "chromium(cdp)"
        session.browser_channel = None
        session.exec_path = None
        session.cdp_url = browser_cdp_url
    else:
        # 在当前环境里直接启动浏览器（本地 / 容器）
        browser_name = (browser_name or "chromium").lower().strip()
        if browser_name not in ("chromium", "firefox", "webkit"):
            logger.warning(f"[{session.session_id}] Unknown browser '{browser_name}', fallback to chromium")
            browser_name = "chromium"

        engine = getattr(_playwright, browser_name)
        launch_args: Dict[str, Any] = dict(headless=bool(headless))

        if browser_name == "chromium":
            if executable_path:
                ep = os.path.normpath(os.path.expandvars(os.path.expanduser(executable_path)))
                if not os.path.isfile(ep):
                    raise ValueError(f"executable_path does not exist: {ep}")
                launch_args["executable_path"] = ep
            elif browser_channel:
                launch_args["channel"] = browser_channel

        logger.info(f"[{session.session_id}] Launching browser: engine={browser_name}, args={launch_args}")
        browser = await engine.launch(**launch_args)
        context = await browser.new_context()
        page = await context.new_page()

        session.browser = browser
        session.context = context
        session.page = page

        session.browser_name = browser_name
        session.browser_channel = browser_channel if browser_name == "chromium" else None
        session.exec_path = launch_args.get("executable_path")
        session.cdp_url = None

    # ---------- 3. 打开 WebUI 页面 ----------
    logger.info(f"[{session.session_id}] 🌐 Opening WebUI page: {webui_url}")
    await page.goto(webui_url)

    # 尽量等页面稳定一点（Gradio 首次渲染会有一波资源加载）
    try:
        await page.wait_for_load_state("networkidle", timeout=60_000)
    except Exception:
        pass
    await asyncio.sleep(settle_delay_ms / 1000.0 if settle_delay_ms else 2.0)

    # ---------- 4. 定义点击 Run Agent 的逻辑（沿用你第一版 http_run.py 的思路） ----------
    async def click_run_agent_once() -> bool:
        selectors = run_tab_selectors or [
            "#component-82-button",
            "button:has-text('Run Agent')",
            "[id*='component'][id$='-button']:has-text('Run Agent')",
            "role=button[name='Run Agent']",
        ]
        for sel in selectors:
            try:
                logger.debug(f"[{session.session_id}] Try click Run Agent: {sel}")
                await page.click(sel, timeout=4000)
                logger.info(f"[{session.session_id}] Clicked Run Agent via: {sel}")
                return True
            except Exception as e:
                logger.debug(f"[{session.session_id}] Selector failed: {sel} err={e}")
        # 再做一层兜底：遍历所有 button 文本找 🤖 Run Agent
        try:
            buttons = page.locator("button")
            count_btn = await buttons.count()
            logger.debug(f"[{session.session_id}] Fallback scanning buttons for 'Run Agent', count={count_btn}")
            for i in range(count_btn):
                try:
                    btn = buttons.nth(i)
                    if not await btn.is_visible():
                        continue
                    text = (await btn.inner_text()).strip()
                    if "Run Agent" in text:
                        await btn.click(timeout=4000)
                        logger.info(f"[{session.session_id}] Fallback clicked button[{i}] with text='{text}'")
                        return True
                except Exception:
                    continue
        except Exception as e:
            logger.debug(f"[{session.session_id}] Fallback scan buttons failed: {e}")
        return False

    # ---------- 5. 循环：检测任务输入框，没有就持续点 Run Agent ----------
    deadline = time.time() + 120  # 最多等 120 秒
    task_textarea = None

    while time.time() < deadline:
        logger.debug(f"[{session.session_id}] Checking for task textarea...")

        tas = page.locator(f"textarea[placeholder='{TASK_PLACEHOLDER}']")
        count_ta = await tas.count()
        visible_handle = None
        for i in range(count_ta):
            h = tas.nth(i)
            try:
                if await h.is_visible():
                    visible_handle = h
                    break
            except Exception:
                continue

        if visible_handle:
            logger.info(f"[{session.session_id}] Task textarea detected, UI ready.")
            task_textarea = visible_handle
            break

        logger.debug(f"[{session.session_id}] Task textarea not found, clicking Run Agent...")
        clicked = await click_run_agent_once()
        if clicked:
            logger.debug(f"[{session.session_id}] Run Agent clicked, waiting UI update...")
        else:
            logger.debug(f"[{session.session_id}] No Run Agent button worked this round.")
        await asyncio.sleep(2.0)

    if task_textarea is None:
        raise RuntimeError("Task textarea not available after clicking Run Agent repeatedly.")

    # ---------- 6. 填写任务 ----------
    try:
        await task_textarea.scroll_into_view_if_needed()
    except Exception:
        pass

    await task_textarea.click()
    await task_textarea.fill(task_text)
    logger.info(f"[{session.session_id}] 📝 Filled task text.")

    # ---------- 7. 查找并点击 ▶️ Submit Task ----------
    logger.info(f"[{session.session_id}] Searching for Submit Task button...")

    buttons = page.locator("button")
    count_btn = await buttons.count()
    submit_button = None

    for i in range(count_btn):
        try:
            btn = buttons.nth(i)
            if not await btn.is_visible():
                continue
            text = (await btn.inner_text()).strip()
            if "Submit Task" in text or "▶️ Submit Task" in text:
                submit_button = btn
                logger.info(f"[{session.session_id}] Found Submit Task button[{i}]: {text}")
                break
        except Exception:
            continue

    if not submit_button:
        raise RuntimeError("Could not find Submit Task button.")

    try:
        await submit_button.scroll_into_view_if_needed()
    except Exception:
        pass

    await submit_button.click(timeout=max_wait_ms)
    logger.info(f"[{session.session_id}] ▶️ Submit Task clicked successfully.")


# -----------------------------
# Click button on current page (Pause/Resume) for session
# -----------------------------
async def _click_button_on_current_page(session: SessionState, button_text: str, max_wait_ms: int = 10_000):
    if session.page is None:
        raise RuntimeError("No connected browser page available for this session; please Start first.")
    await session.page.click(f"button:has-text('{button_text}')", timeout=max_wait_ms)
    logger.info(f"[{session.session_id}] ✅ Clicked button: {button_text}")

# -----------------------------
# Close browser (Stop) for session
# -----------------------------
async def _close_browser_for_session(session: SessionState):
    if session.browser is not None:
        try:
            logger.info(f"[{session.session_id}] Closing browser...")
            await session.browser.close()
        except Exception as e:
            logger.warning(f"[{session.session_id}] Error closing browser: {e}")
    session.browser = session.context = session.page = None

# -----------------------------
# Background event loop
# -----------------------------
class AsyncWorker:
    def __init__(self):
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def _run_loop(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def submit_coro(self, coro: Awaitable):
        if self.loop is None:
            raise RuntimeError("AsyncWorker event loop not ready.")
        return asyncio.run_coroutine_threadsafe(coro, self.loop)

# -----------------------------
# FastAPI
# -----------------------------
app = FastAPI(title="UI Task Runner", version="3.1.0")
worker = AsyncWorker()

class ActionReq(BaseModel):
    Action: str                  # "Start" | "Pause" | "Resume" | "Stop"
    session_id: Optional[str] = None   # 每个任务/会话的 ID
    task: Optional[str] = None   # Required for Start
    cdp_URP: Optional[str] = None
    # LLM configuration
    llm_model_name: Optional[str] = None
    llm_base_url: Optional[str] = None

    host: Optional[str] = None
    port: Optional[int] = None
    headless: Optional[bool] = False
    run_tab_selectors: Optional[List[str]] = None   # 当前未使用，仅兼容
    max_wait_ms: Optional[int] = 15_000             # 当前未使用，仅兼容
    settle_delay_ms: Optional[int] = 2_000          # 当前未使用，仅兼容

    # External browser selection
    browser: Optional[str] = "chromium"           # "chromium" | "firefox" | "webkit"
    browser_channel: Optional[str] = None         # Only for chromium
    executable_path: Optional[str] = None         # Only for chromium

    # Connect to external browser directly via CDP（本版本未使用，仅兼容字段）
    browser_cdp__url: Optional[str] = None

class ActionResp(BaseModel):
    status: str
    message: str
    session_id: Optional[str] = None
    webui_url: Optional[str] = None
    webui_ws_url: Optional[str] = None    # ✅ 新增：对应的 WebSocket URL
    pid: Optional[int] = None
    browser: Optional[str] = None
    browser_channel: Optional[str] = None
    executable_path: Optional[str] = None
    browser_cdp__url: Optional[str] = None

@app.get("/health")
def health():
    with _sessions_lock:
        sessions_info = []
        for sid, s in _sessions.items():
            sessions_info.append(
                {
                    "session_id": sid,
                    "host": s.host,
                    "port": s.port,
                    "ws_port": s.ws_port,  # ✅ 新增：WS 端口
                    "webui_url": f"http://{s.host}:{s.port}" if s.port else None,
                    "webui_ws_url": f"ws://{s.host}:{s.ws_port}" if s.ws_port else None,
                    "webui_pid": s.webui_proc.pid if s.webui_proc else None,
                    "browser_open": bool(s.browser is not None),
                    "browser": s.browser_name,
                    "browser_channel": s.browser_channel,
                    "executable_path": s.exec_path,
                    "browser_cdp__url": s.cdp_url,
                }
            )
    return {
        "status": "ok",
        "script": WEBUI_SCRIPT_PATH,
        "json": WEBUI_JSON_PATH,
        "sessions": sessions_info,
    }

@app.get("/version")
def version():
    return {"version": app.version}

@app.post("/action", response_model=ActionResp)
def handle_action(req: ActionReq):
    action = (req.Action or "").strip().lower()
    logger.info(f"Received /action: {action}")
    logger.debug(f"Request body: {req.model_dump_json()}")

    if action not in ("start", "pause", "resume", "stop"):
        raise HTTPException(status_code=400, detail=f"Unknown Action: {req.Action}")

    # ---------- Start ----------
    if action == "start":
        host = (req.host or DEFAULT_WEBUI_HOST).strip()
        try:
            port = int(req.port) if req.port else DEFAULT_WEBUI_PORT
        except Exception:
            port = DEFAULT_WEBUI_PORT

        if not req.task or not req.task.strip():
            raise HTTPException(status_code=400, detail="Start action requires a non-empty task")

        # 生成/使用 session_id
        session_id = req.session_id or uuid.uuid4().hex

        try:
            session = create_session(session_id, host, port)
        except ValueError:
            raise HTTPException(status_code=409, detail=f"Session already exists: {session_id}")

        async def _job_start():
            # 1) 写 webui.json
            write_or_update_webui_json(
                req.cdp_URP,
                llm_model_name=req.llm_model_name,
                llm_base_url=req.llm_base_url,
            )

            # 2) 启动当前 session 的 webui.py
            def _start_proc():
                return launch_webui_process_for_session(session)

            proc, real_port = await asyncio.get_event_loop().run_in_executor(None, _start_proc)
            webui_url = f"http://{session.host}:{real_port}"

            # 3) 打开浏览器（支持 CDP 外部浏览器）并提交任务
            await _open_browser_and_submit_for_session(
                session=session,
                webui_url=webui_url,
                task_text=req.task.strip(),
                run_tab_selectors=req.run_tab_selectors,
                headless=bool(req.headless),
                max_wait_ms=int(req.max_wait_ms or 15000),
                settle_delay_ms=int(req.settle_delay_ms or 2000),
                browser_name=(req.browser or "chromium"),
                browser_channel=req.browser_channel,
                executable_path=req.executable_path,
                browser_cdp_url=req.browser_cdp__url,
            )

            return webui_url, proc.pid, session

        try:
            fut = worker.submit_coro(_job_start())
            webui_url, pid, session = fut.result()
            msg = f"Start accepted for session={session_id}, WebUI: {webui_url}. Browser remains open."
            if req.cdp_URP is not None:
                msg += f" Updated cdp_URP in {WEBUI_JSON_PATH}."
            if req.llm_model_name is not None:
                msg += f" Updated llm_model_name={req.llm_model_name}."
            if req.llm_base_url is not None:
                msg += f" Updated llm_base_url={req.llm_base_url}."
            ws_url = f"ws://{session.host}:{session.ws_port}" if session.ws_port else None
            return ActionResp(
                status="accepted",
                message=msg,
                session_id=session_id,
                webui_url=webui_url,
                webui_ws_url=ws_url,
                pid=pid,
                browser=session.browser_name,
                browser_channel=session.browser_channel,
                executable_path=session.exec_path,
                browser_cdp__url=session.cdp_url,
            )
        except HTTPException:
            remove_session(session_id)
            raise
        except Exception as e:
            logger.exception("Start execution failed")
            remove_session(session_id)
            raise HTTPException(status_code=500, detail=f"Start execution failed: {e}")

    # ---------- Pause / Resume ----------
    if action in ("pause", "resume"):
        if not req.session_id:
            raise HTTPException(status_code=400, detail="Pause/Resume requires session_id")
        try:
            session = get_session(req.session_id)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"Session not found: {req.session_id}")

        btn = "Pause" if action == "pause" else "Resume"

        async def _job_click():
            await _click_button_on_current_page(session, button_text=btn)

        try:
            resume_req_ts = time.time()
            fut = worker.submit_coro(_job_click())
            fut.result()

            extra = ""
            if action == "resume":
                # 清空事件
                with session.enter_prompt_lock:
                    session.enter_prompt_event.clear()
                session.enter_consumed_event.clear()

                # A) 等待 CLI 提示
                got_prompt = session.enter_prompt_event.wait(timeout=RESUME_WAIT_FOR_PROMPT_SECONDS)
                if got_prompt:
                    session.enter_consumed_event.wait(timeout=RESUME_GRACE_FOR_CONSUME_SECONDS)
                    if session.enter_consumed_event.is_set():
                        extra = " (detected prompt → ENTER already consumed)"
                        with session.enter_prompt_lock:
                            session.enter_prompt_event.clear()
                        session.enter_consumed_event.clear()
                    else:
                        ok = send_enter_to_webui(session)
                        extra = " (detected prompt → ENTER x1)" if ok else " (prompt detected but send failed)"
                        with session.enter_prompt_lock:
                            session.enter_prompt_event.clear()
                else:
                    # 没有提示，也给一个小观察窗口
                    session.enter_consumed_event.wait(timeout=RESUME_GRACE_FOR_CONSUME_SECONDS)
                    if session.enter_consumed_event.is_set():
                        extra = " (no prompt, but ENTER already consumed)"
                        session.enter_consumed_event.clear()
                    else:
                        if RESUME_FALLBACK_SEND_ENTER:
                            ok = send_enter_to_webui(session)
                            extra = " (no prompt within timeout → fallback ENTER x1)" if ok else " (fallback send failed)"
                        else:
                            extra = " (no prompt; no ENTER sent)"

                # B) 观察是否立即又被 Pause，一次自动恢复
                time.sleep(RESUME_OBSERVE_AFTER_SECONDS)
                if session.last_pause_ts > resume_req_ts:
                    logger.warning(f"[{session.session_id}] Detected immediate re-pause after resume. Performing one auto-recover resume.")
                    fut2 = worker.submit_coro(_job_click())
                    fut2.result()

                    with session.enter_prompt_lock:
                        session.enter_prompt_event.clear()
                    session.enter_consumed_event.clear()

                    got_prompt2 = session.enter_prompt_event.wait(timeout=RESUME_WAIT_FOR_PROMPT_SECONDS)
                    if got_prompt2:
                        session.enter_consumed_event.wait(timeout=RESUME_GRACE_FOR_CONSUME_SECONDS)
                        if session.enter_consumed_event.is_set():
                            extra += " | auto-recover: ENTER already consumed"
                            with session.enter_prompt_lock:
                                session.enter_prompt_event.clear()
                            session.enter_consumed_event.clear()
                        else:
                            ok2 = send_enter_to_webui(session)
                            extra += " | auto-recover: ENTER x1" if ok2 else " | auto-recover: send failed"
                            with session.enter_prompt_lock:
                                session.enter_prompt_event.clear()
                    else:
                        session.enter_consumed_event.wait(timeout=RESUME_GRACE_FOR_CONSUME_SECONDS)
                        if session.enter_consumed_event.is_set():
                            extra += " | auto-recover: ENTER already consumed (no prompt)"
                            session.enter_consumed_event.clear()
                        else:
                            if RESUME_FALLBACK_SEND_ENTER:
                                ok2 = send_enter_to_webui(session)
                                extra += " | auto-recover: fallback ENTER x1" if ok2 else " | auto-recover: fallback failed"
                            else:
                                extra += " | auto-recover: no prompt; no ENTER sent"

            webui_url = f"http://{session.host}:{session.port}"
            ws_url = f"ws://{session.host}:{session.ws_port}" if session.ws_port else None
            msg = f"{req.Action} accepted for session={session.session_id} (clicked {btn}).{extra}"
            return ActionResp(
                status="accepted",
                message=msg,
                session_id=session.session_id,
                webui_url=webui_url,
                webui_ws_url=ws_url,
                browser=session.browser_name,
                browser_channel=session.browser_channel,
                executable_path=session.exec_path,
                browser_cdp__url=session.cdp_url,
            )
        except RuntimeError as e:
            raise HTTPException(status_code=409, detail=f"{req.Action} failed: {e}")
        except Exception as e:
            logger.exception(f"{req.Action} execution failed")
            raise HTTPException(status_code=500, detail=f"{req.Action} execution failed: {e}")

    # ---------- Stop ----------
    if action == "stop":
        if not req.session_id:
            raise HTTPException(status_code=400, detail="Stop requires session_id")
        try:
            session = get_session(req.session_id)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"Session not found: {req.session_id}")

        async def _job_stop():
            # 1) Close browser
            await _close_browser_for_session(session)

            # 2) Stop webui.py by handle
            proc = session.webui_proc
            if proc is not None:
                try:
                    if psutil:
                        ps_proc = psutil.Process(proc.pid)
                        ps_proc.terminate()
                        try:
                            ps_proc.wait(timeout=2)
                        except Exception:
                            ps_proc.kill()
                    else:
                        if os.name == "nt":
                            subprocess.run(
                                ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL,
                            )
                        else:
                            import signal
                            os.kill(proc.pid, signal.SIGTERM)
                            time.sleep(0.3)
                            os.kill(proc.pid, signal.SIGKILL)
                    logger.info(f"[{session.session_id}] Stopped webui.py by handle (pid={proc.pid}).")
                except Exception as e:
                    logger.warning(f"[{session.session_id}] Failed to stop webui.py by handle: {e}")

            session.webui_proc = None

        try:
            fut = worker.submit_coro(_job_stop())
            fut.result()
            remove_session(session.session_id)
            return ActionResp(
                status="ok",
                message=f"Stop completed for session={session.session_id}: closed browser and terminated webui.py.",
                session_id=session.session_id,
            )
        except Exception as e:
            logger.exception("Stop execution failed")
            raise HTTPException(status_code=500, detail=f"Stop execution failed: {e}")

    raise HTTPException(status_code=400, detail=f"Unknown Action: {req.Action}")

# -----------------------------
# Entry point: Start API only (do not start WebUI)
# -----------------------------
def start_api_server(host: str = None, port: int = None):
    api_host = host or os.environ.get("API_HOST", "0.0.0.0")
    api_port = int(port or os.environ.get("API_PORT", "9000"))
    logger.info(f"🛰️ Starting API server: http://{api_host}:{api_port}")
    uvicorn.run(app, host=api_host, port=api_port, log_level="debug")

if __name__ == "__main__":
    start_api_server()
