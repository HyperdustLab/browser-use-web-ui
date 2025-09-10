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
from collections import deque

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

# ====== Optional pid file for managing webui.py instances started by this script ======
PID_DIR = os.path.join(BASE_DIR, "tmp")
PID_FILE = os.path.join(PID_DIR, "webui.pid")


WEBUI_PROC: Optional[subprocess.Popen] = None

# ====== Current WebUI and browser status ======
CURRENT_HOST: Optional[str] = None
CURRENT_PORT: Optional[int] = None
CURRENT_BROWSER_NAME: Optional[str] = None
CURRENT_BROWSER_CHANNEL: Optional[str] = None
CURRENT_EXEC_PATH: Optional[str] = None
CURRENT_CDP_URL: Optional[str] = None
_current_lock = threading.Lock()

# ====== Global Playwright state (reusing browser/page across requests) ======
_playwright = None
_browser = None
_context = None
_page = None


ENTER_PROMPT_SUBSTR = "Press [Enter] to resume"
ENTER_PROMPT_EVENT = threading.Event()
ENTER_PROMPT_LOCK = threading.Lock()
ENTER_PROMPT_LAST_TS = 0.0


LAST_PAUSE_TS = 0.0
ENTER_CONSUMED_EVENT = threading.Event()
ENTER_CONSUMED_LAST_TS = 0.0


LAST_LINES = deque(maxlen=200)


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
    data = {}
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
# Process management: Kill webui.py
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
                subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                import signal
                os.kill(pid, signal.SIGTERM)
                time.sleep(0.5)
                os.kill(pid, signal.SIGKILL)
        except Exception:
            pass

def _pids_listening_on_port(port: int) -> List[int]:
    pids = []
    if psutil:
        try:
            for c in psutil.net_connections(kind="inet"):
                if c.laddr and c.laddr.port == port and c.status == psutil.CONN_LISTEN and c.pid:
                    pids.append(c.pid)
        except Exception:
            pass
    else:
        try:
            if os.name == "nt":
                out = subprocess.check_output(["netstat", "-ano"], text=True, creationflags=subprocess.CREATE_NO_WINDOW)
                for line in out.splitlines():
                    if f":{port} " in line and "LISTENING" in line:
                        parts = line.split()
                        pids.append(int(parts[-1]))
        except Exception:
            pass
    return list(sorted(set(pids)))

def kill_existing_webui(host: str, port: int):
    logger.info("🛑 Attempting to stop existing webui.py.")
    if os.path.exists(PID_FILE):
        try:
            with open(PID_FILE, "r", encoding="utf-8") as f:
                pid = int(f.read().strip())
            _kill_pid(pid)
        except Exception:
            pass
        try:
            os.remove(PID_FILE)
        except Exception:
            pass
    if psutil:
        try:
            for p in psutil.process_iter(attrs=["pid", "name", "cmdline"]):
                try:
                    cmdline = " ".join(p.info.get("cmdline") or [])
                    if "webui.py" in cmdline:
                        _kill_pid(p.info["pid"])
                except Exception:
                    continue
        except Exception:
            pass
    for pid in _pids_listening_on_port(port):
        _kill_pid(pid, force=True)
    for _ in range(20):  # Wait up to ~4s
        if is_port_free(host, port):
            break
        time.sleep(0.2)

# -----------------------------
# Subprocess output handling (with prompt/pause/enter consumption matching)
# -----------------------------
def _pump_output(proc: subprocess.Popen, prefix: str):
    def _reader(stream):
        global ENTER_PROMPT_LAST_TS, LAST_PAUSE_TS, ENTER_CONSUMED_LAST_TS
        for line in iter(stream.readline, b""):
            try:
                text = line.decode(errors='ignore')
            except Exception:
                text = ""

            # Print
            try:
                sys.stdout.write(f"{prefix}{text}")
                sys.stdout.flush()
            except Exception:
                pass

            # Record recent lines
            if text:
                LAST_LINES.append(text.rstrip("\n"))

            # —— 1) CLI prompts "Press [Enter] to resume ..."
            if ENTER_PROMPT_SUBSTR in text:
                with ENTER_PROMPT_LOCK:
                    ENTER_PROMPT_LAST_TS = time.time()
                    ENTER_PROMPT_EVENT.set()
                logger.debug("Detected CLI resume prompt; event set.")

            # —— 2) UI/CLI pause traces
            if ("Pause button clicked." in text) or ("Got Ctrl+C, paused" in text):
                LAST_PAUSE_TS = time.time()

            # —— 3) CLI prints "Got Enter, resuming ..."
            if "Got Enter, resuming" in text:
                ENTER_CONSUMED_LAST_TS = time.time()
                ENTER_CONSUMED_EVENT.set()
                logger.debug("Detected ENTER already consumed by CLI.")

    if proc.stdout:
        t = threading.Thread(target=_reader, args=(proc.stdout,), daemon=True)
        t.start()

def _watch_exit(proc: subprocess.Popen, label: str):
    def _waiter():
        code = proc.wait()
        logger.warning(f"{label} exited with returncode={code}")
    threading.Thread(target=_waiter, daemon=True).start()

# -----------------------------
# Send "Enter" to webui.py process
# -----------------------------
def send_enter_to_webui() -> bool:
    """Write a newline to webui.py stdin to simulate pressing Enter in CLI."""
    global WEBUI_PROC
    if WEBUI_PROC is None:
        logger.warning("send_enter_to_webui: WEBUI_PROC is None (webui not started by this process).")
        return False
    try:
        if WEBUI_PROC.stdin:
            WEBUI_PROC.stdin.write(b"\n")
            WEBUI_PROC.stdin.flush()
            logger.info("▶️  Sent ENTER to webui.py stdin.")
            return True
        else:
            logger.warning("send_enter_to_webui: WEBUI_PROC.stdin is None.")
            return False
    except Exception as e:
        logger.exception(f"send_enter_to_webui failed: {e}")
        return False

# -----------------------------
# Launch webui.py
# -----------------------------
def launch_webui_process(host: str, port: int):
    chosen_port = port
    if not is_port_free(host, chosen_port):
        logger.warning(f"Port {chosen_port} is occupied, automatically selecting a free port")
        chosen_port = find_free_port(host)

    ensure_dir_for(PID_FILE)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    # Read llm_base_url from webui.json → inject into child process OLLAMA_HOST (if OAI reverse proxy, automatically remove /v1)
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
                logger.info(f"Set OLLAMA_HOST={env['OLLAMA_HOST']} for subprocess")
        else:
            logger.info(f"webui.json not found ({WEBUI_JSON_PATH}), skipping OLLAMA_HOST injection")
    except Exception as e:
        logger.warning(f"Failed to read webui.json to set OLLAMA_HOST: {e}")

    cmd = [PYTHON_EXE, WEBUI_SCRIPT_PATH, "--ip", host, "--port", str(chosen_port)]
    logger.info("🚀 Launching webui.py: %s", " ".join(cmd))
    proc = subprocess.Popen(
        cmd,
        cwd=BASE_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        stdin=subprocess.PIPE,          # ✅ Enable stdin for Resume to write Enter
        env=env,
        creationflags=0
    )

    # Save to global handle
    global WEBUI_PROC
    WEBUI_PROC = proc

    _pump_output(proc, "[webui.py] ")
    _watch_exit(proc, "webui.py")

    try:
        with open(PID_FILE, "w", encoding="utf-8") as f:
            f.write(str(proc.pid))
    except Exception:
        pass

    root_url = f"http://{host}:{chosen_port}/"
    ok = wait_http_ok(root_url, timeout_s=60.0, interval_s=1.0)  # ← 60s/1s
    if ok:
        logger.info("webui.py health check successful: %s", root_url)
    else:
        logger.warning("webui.py health check not successful (may become ready later)")

    global CURRENT_HOST, CURRENT_PORT
    with _current_lock:
        CURRENT_HOST, CURRENT_PORT = host, chosen_port

    return proc, chosen_port

# -----------------------------
# Playwright: Launch/connect to browser & submit task
# -----------------------------
async def _open_fresh_browser_and_submit(
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
    - Prefer connect_over_cdp if browser_cdp_url provided
    - Otherwise launch specified engine (chromium|firefox|webkit)
    - Open WebUI → Run Agent → fill textarea → Submit Task
    - Keep browser open for Pause/Resume
    """
    global _playwright, _browser, _context, _page
    global CURRENT_BROWSER_NAME, CURRENT_BROWSER_CHANNEL, CURRENT_EXEC_PATH, CURRENT_CDP_URL

    # Close existing browser before new task
    if _browser is not None:
        try:
            logger.info("Closing existing browser (before starting new task)...")
            await _browser.close()
        except Exception as e:
            logger.warning(f"Error closing old browser: {e}")
        finally:
            _browser = _context = _page = None

    if _playwright is None:
        _playwright = await async_playwright().start()

    using_cdp = False
    if browser_cdp_url:
        logger.info(f"Connecting to external browser via CDP: {browser_cdp_url}")
        _browser = await _playwright.chromium.connect_over_cdp(browser_cdp_url)
        _context = _browser.contexts[0] if _browser.contexts else await _browser.new_context()
        _page = await _context.new_page()
        using_cdp = True
        with _current_lock:
            CURRENT_BROWSER_NAME = "chromium(cdp)"
            CURRENT_BROWSER_CHANNEL = None
            CURRENT_EXEC_PATH = None
            CURRENT_CDP_URL = browser_cdp_url
    else:
        # Normal launch
        browser_name = (browser_name or "chromium").lower().strip()
        if browser_name not in ("chromium", "firefox", "webkit"):
            logger.warning(f"Unknown browser '{browser_name}', using chromium instead")
            browser_name = "chromium"
        engine = getattr(_playwright, browser_name)
        launch_args = dict(headless=headless)
        if browser_name == "chromium":
            if executable_path:
                ep = os.path.normpath(os.path.expandvars(os.path.expanduser(executable_path)))
                if not os.path.isfile(ep):
                    raise ValueError(f"executable_path does not exist: {ep}")
                launch_args["executable_path"] = ep
            elif browser_channel:
                launch_args["channel"] = browser_channel
        else:
            if browser_channel:
                logger.warning(f"{browser_name} does not support browser_channel, ignored")
            if executable_path:
                logger.warning(f"{browser_name} does not support executable_path, ignored")
        logger.info(f"Launching browser: engine={browser_name}, args={ {k:v for k,v in launch_args.items() if v} }")
        _browser = await engine.launch(**launch_args)
        _context = await _browser.new_context()
        _page = await _context.new_page()
        with _current_lock:
            CURRENT_BROWSER_NAME = browser_name
            CURRENT_BROWSER_CHANNEL = browser_channel if browser_name == "chromium" else None
            CURRENT_EXEC_PATH = launch_args.get("executable_path")
            CURRENT_CDP_URL = None

    logger.info(f"🌐 Opening WebUI page: {webui_url}")
    await _page.goto(webui_url, wait_until="domcontentloaded")

    # Click Run Agent tab
    selectors = run_tab_selectors or [
        "#component-82-button",
        "button:has-text('Run Agent')",
        "[id*='component'][id$='-button']:has-text('Run Agent')",
        "role=button[name='Run Agent']",
    ]
    if settle_delay_ms > 0:
        await asyncio.sleep(settle_delay_ms / 1000.0)
    for sel in selectors:
        try:
            logger.debug(f"Attempting to click tab: {sel}")
            await _page.click(sel, timeout=4000)
            break
        except Exception:
            pass

    # Visible textarea
    try:
        await _page.wait_for_selector("textarea", timeout=max_wait_ms, state="attached")
    except Exception as e:
        raise RuntimeError(f"No textarea found on page: {e}")

    textareas = _page.locator("textarea")
    count = await textareas.count()
    target = None
    for i in range(min(count, 25)):
        handle = textareas.nth(i)
        try:
            if await handle.is_visible():
                target = handle
                break
        except Exception:
            continue
    if target is None:
        target = textareas.first
    try:
        await target.scroll_into_view_if_needed(timeout=3000)
    except Exception:
        pass

    # Fill task
    try:
        await target.fill(task_text, timeout=max_wait_ms)
    except Exception:
        try:
            await target.click(timeout=3000)
            await _page.keyboard.type(task_text, delay=20)
        except Exception as e:
            raise RuntimeError(f"Failed to fill task into textarea: {e}")

    # Submit
    try:
        await _page.click("button:has-text('Submit Task')", timeout=max_wait_ms)
    except Exception:
        buttons = _page.locator("button")
        bcount = await buttons.count()
        clicked = False
        for i in range(min(bcount, 30)):
            b = buttons.nth(i)
            try:
                if await b.is_visible():
                    await b.click()
                    clicked = True
                    break
            except Exception:
                continue
        if not clicked:
            raise RuntimeError("No clickable button found on page (cannot submit task).")

    logger.info("✅ Task submitted (browser remains open for Pause/Resume)")

    # CDP mode: Clean up extra tabs
    if using_cdp:
        try:
            for ctx in list(_browser.contexts):
                for pg in list(ctx.pages):
                    try:
                        if pg != _page and not pg.is_closed():
                            await pg.close()
                    except Exception:
                        continue
            logger.info("🔧 Closed extra tabs, keeping only current page.")
        except Exception as e:
            logger.warning(f"Failed to clean up extra tabs: {e}")

# -----------------------------
# Click button on current page (Pause/Resume)
# -----------------------------
async def _click_button_on_current_page(button_text: str, max_wait_ms: int = 10_000):
    global _page
    if _page is None:
        raise RuntimeError("No connected browser page available; please Start first.")
    await _page.click(f"button:has-text('{button_text}')", timeout=max_wait_ms)
    logger.info(f"✅ Clicked button: {button_text}")

# -----------------------------
# Close browser (Stop)
# -----------------------------
async def _close_browser_if_any():
    global _page, _context, _browser
    if _browser is not None:
        try:
            logger.info("Closing browser...")
            await _browser.close()
        except Exception as e:
            logger.warning(f"Error closing browser: {e}")
    _page = _context = _browser = None

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
app = FastAPI(title="UI Task Runner", version="2.11.0")
worker = AsyncWorker()
_task_lock = asyncio.Lock()
_proc_lock = threading.Lock()

class ActionReq(BaseModel):
    Action: str                  # "Start" | "Pause" | "Resume" | "Stop"
    task: Optional[str] = None   # Required for Start
    cdp_URP: Optional[str] = None
    # LLM configuration
    llm_model_name: Optional[str] = None
    llm_base_url: Optional[str] = None

    host: Optional[str] = None
    port: Optional[int] = None
    headless: Optional[bool] = False
    run_tab_selectors: Optional[List[str]] = None
    max_wait_ms: Optional[int] = 15_000
    settle_delay_ms: Optional[int] = 2_000

    # External browser selection
    browser: Optional[str] = "chromium"           # "chromium" | "firefox" | "webkit"
    browser_channel: Optional[str] = None         # Only for chromium
    executable_path: Optional[str] = None         # Only for chromium

    # Connect to external browser directly via CDP
    browser_cdp__url: Optional[str] = None

class ActionResp(BaseModel):
    status: str
    message: str
    webui_url: Optional[str] = None
    pid: Optional[int] = None
    browser: Optional[str] = None
    browser_channel: Optional[str] = None
    executable_path: Optional[str] = None
    browser_cdp__url: Optional[str] = None

@app.get("/health")
def health():
    with _current_lock:
        ch, cp = CURRENT_HOST, CURRENT_PORT
        bn, bc, ep, cu = CURRENT_BROWSER_NAME, CURRENT_BROWSER_CHANNEL, CURRENT_EXEC_PATH, CURRENT_CDP_URL
    return {
        "status": "ok",
        "script": WEBUI_SCRIPT_PATH,
        "json": WEBUI_JSON_PATH,
        "current_host": ch,
        "current_port": cp,
        "browser_open": bool(_browser is not None),
        "browser": bn,
        "browser_channel": bc,
        "executable_path": ep,
        "browser_cdp__url": cu,
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

        async def _job_start():
            async with _task_lock:
                # Close old browser
                await _close_browser_if_any()

                # Kill old webui
                def _stop_proc():
                    with _proc_lock:
                        kill_existing_webui(host, port)
                await asyncio.get_event_loop().run_in_executor(None, _stop_proc)

                # Write json
                write_or_update_webui_json(
                    req.cdp_URP,
                    llm_model_name=req.llm_model_name,
                    llm_base_url=req.llm_base_url,
                )

                # Start webui
                def _start_proc():
                    with _proc_lock:
                        return launch_webui_process(host, port)
                proc, real_port = await asyncio.get_event_loop().run_in_executor(None, _start_proc)
                webui_url = f"http://{host}:{real_port}"

                # Open browser & submit
                await _open_fresh_browser_and_submit(
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

                with _current_lock:
                    bn, bc, ep, cu = CURRENT_BROWSER_NAME, CURRENT_BROWSER_CHANNEL, CURRENT_EXEC_PATH, CURRENT_CDP_URL

                return webui_url, proc.pid, bn, bc, ep, cu

        try:
            fut = worker.submit_coro(_job_start())
            webui_url, pid, bn, bc, ep, cu = fut.result()
            msg = f"Start accepted, WebUI: {webui_url}. Browser remains open (engine={bn}"
            if bc:
                msg += f", channel={bc}"
            if ep:
                msg += f", exec={ep}"
            if cu:
                msg += f", cdp={cu}"
            msg += ")."
            if req.cdp_URP is not None:
                msg += f" Updated cdp_URP in {WEBUI_JSON_PATH}."
            if req.llm_model_name is not None:
                msg += f" Updated llm_model_name={req.llm_model_name}."
            if req.llm_base_url is not None:
                msg += f" Updated llm_base_url={req.llm_base_url}."
            return ActionResp(
                status="accepted",
                message=msg,
                webui_url=webui_url,
                pid=pid,
                browser=bn,
                browser_channel=bc,
                executable_path=ep,
                browser_cdp__url=cu,
            )
        except HTTPException:
            raise
        except Exception as e:
            logger.exception("Start execution failed")
            raise HTTPException(status_code=500, detail=f"Start execution failed: {e}")

    # ---------- Pause / Resume ----------
    if action in ("pause", "resume"):
        btn = "Pause" if action == "pause" else "Resume"

        async def _job_click_resume_or_pause():
            async with _task_lock:
                await _click_button_on_current_page(button_text=btn)

        try:
            # Record request timestamp (to determine if immediately re-paused)
            resume_req_ts = time.time()

            # 1) First click UI's Pause/Resume
            fut = worker.submit_coro(_job_click_resume_or_pause())
            fut.result()

            extra = ""
            if action == "resume":
                # Clear events
                with ENTER_PROMPT_LOCK:
                    ENTER_PROMPT_EVENT.clear()
                ENTER_CONSUMED_EVENT.clear()

                # A) Wait for prompt
                got_prompt = ENTER_PROMPT_EVENT.wait(timeout=RESUME_WAIT_FOR_PROMPT_SECONDS)

                if got_prompt:
                    # Give a short observation window to see if any "stale Enter" was consumed
                    ENTER_CONSUMED_EVENT.wait(timeout=RESUME_GRACE_FOR_CONSUME_SECONDS)
                    if ENTER_CONSUMED_EVENT.is_set():
                        # Already consumed, no new Enter sent
                        extra = " (detected prompt → ENTER already consumed)"
                        with ENTER_PROMPT_LOCK:
                            ENTER_PROMPT_EVENT.clear()
                        ENTER_CONSUMED_EVENT.clear()
                    else:
                        ok = send_enter_to_webui()
                        extra = " (detected prompt → ENTER x1)" if ok else " (prompt detected but send failed)"
                        with ENTER_PROMPT_LOCK:
                            ENTER_PROMPT_EVENT.clear()
                else:
                    # No prompt received: also give short observation window, don't fallback if consumed
                    ENTER_CONSUMED_EVENT.wait(timeout=RESUME_GRACE_FOR_CONSUME_SECONDS)
                    if ENTER_CONSUMED_EVENT.is_set():
                        extra = " (no prompt, but ENTER already consumed)"
                        ENTER_CONSUMED_EVENT.clear()
                    else:
                        if RESUME_FALLBACK_SEND_ENTER:
                            ok = send_enter_to_webui()
                            extra = " (no prompt within timeout → fallback ENTER x1)" if ok else " (fallback send failed)"
                        else:
                            extra = " (no prompt; no ENTER sent)"

                # B) Short observation after resume: if immediately paused again by UI → auto-recover with another Resume (using "already consumed" check)
                time.sleep(RESUME_OBSERVE_AFTER_SECONDS)
                if LAST_PAUSE_TS > resume_req_ts:
                    logger.warning("Detected immediate re-pause after resume. Performing one auto-recover resume.")
                    fut2 = worker.submit_coro(_job_click_resume_or_pause())  # Still Resume
                    fut2.result()

                    with ENTER_PROMPT_LOCK:
                        ENTER_PROMPT_EVENT.clear()
                    ENTER_CONSUMED_EVENT.clear()

                    got_prompt2 = ENTER_PROMPT_EVENT.wait(timeout=RESUME_WAIT_FOR_PROMPT_SECONDS)
                    if got_prompt2:
                        ENTER_CONSUMED_EVENT.wait(timeout=RESUME_GRACE_FOR_CONSUME_SECONDS)
                        if ENTER_CONSUMED_EVENT.is_set():
                            extra += " | auto-recover: ENTER already consumed"
                            with ENTER_PROMPT_LOCK:
                                ENTER_PROMPT_EVENT.clear()
                            ENTER_CONSUMED_EVENT.clear()
                        else:
                            ok2 = send_enter_to_webui()
                            extra += " | auto-recover: ENTER x1" if ok2 else " | auto-recover: send failed"
                            with ENTER_PROMPT_LOCK:
                                ENTER_PROMPT_EVENT.clear()
                    else:
                        ENTER_CONSUMED_EVENT.wait(timeout=RESUME_GRACE_FOR_CONSUME_SECONDS)
                        if ENTER_CONSUMED_EVENT.is_set():
                            extra += " | auto-recover: ENTER already consumed (no prompt)"
                            ENTER_CONSUMED_EVENT.clear()
                        else:
                            if RESUME_FALLBACK_SEND_ENTER:
                                ok2 = send_enter_to_webui()
                                extra += " | auto-recover: fallback ENTER x1" if ok2 else " | auto-recover: fallback failed"
                            else:
                                extra += " | auto-recover: no prompt; no ENTER sent"

            with _current_lock:
                ch, cp = CURRENT_HOST, CURRENT_PORT
                bn, bc, ep, cu = CURRENT_BROWSER_NAME, CURRENT_BROWSER_CHANNEL, CURRENT_EXEC_PATH, CURRENT_CDP_URL
            webui_url = f"http://{ch or DEFAULT_WEBUI_HOST}:{cp or DEFAULT_WEBUI_PORT}"
            msg = f"{req.Action} accepted (clicked {btn} on current page).{extra}"
            return ActionResp(
                status="accepted",
                message=msg,
                webui_url=webui_url,
                browser=bn,
                browser_channel=bc,
                executable_path=ep,
                browser_cdp__url=cu,
            )
        except RuntimeError as e:
            raise HTTPException(status_code=409, detail=f"{req.Action} failed: {e}")
        except Exception as e:
            logger.exception(f"{req.Action} execution failed")
            raise HTTPException(status_code=500, detail=f"{req.Action} execution failed: {e}")

    # ---------- Stop ----------
    if action == "stop":
        with _current_lock:
            ch, cp = CURRENT_HOST, CURRENT_PORT
        host = (req.host or ch or DEFAULT_WEBUI_HOST).strip()
        try:
            port = int(req.port) if req.port else (cp or DEFAULT_WEBUI_PORT)
        except Exception:
            port = cp or DEFAULT_WEBUI_PORT

        async def _job_stop():
            async with _task_lock:
                # 1) Close browser
                await _close_browser_if_any()

                # 2) Try to stop webui.py we started via handle
                global WEBUI_PROC
                try:
                    if WEBUI_PROC is not None:
                        p = WEBUI_PROC
                        WEBUI_PROC = None
                        try:
                            if psutil:
                                ps_proc = psutil.Process(p.pid)
                                ps_proc.terminate()
                                try:
                                    ps_proc.wait(timeout=2)
                                except Exception:
                                    ps_proc.kill()
                            else:
                                if os.name == "nt":
                                    subprocess.run(["taskkill", "/PID", str(p.pid), "/T", "/F"],
                                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                                else:
                                    import signal
                                    os.kill(p.pid, signal.SIGTERM)
                                    time.sleep(0.3)
                                    os.kill(p.pid, signal.SIGKILL)
                            logger.info(f"Stopped webui.py by handle (pid={p.pid}).")
                        except Exception as e:
                            logger.warning(f"Failed to stop webui.py by handle: {e}")
                except Exception:
                    pass

                # 3) Fallback: Scan/kill by port
                def _stop_proc():
                    with _proc_lock:
                        kill_existing_webui(host, port)
                await asyncio.get_event_loop().run_in_executor(None, _stop_proc)

                # 4) Clean up state
                global CURRENT_HOST, CURRENT_PORT, CURRENT_BROWSER_NAME, CURRENT_BROWSER_CHANNEL, CURRENT_EXEC_PATH, CURRENT_CDP_URL
                with _current_lock:
                    CURRENT_HOST = CURRENT_PORT = None
                    CURRENT_BROWSER_NAME = CURRENT_BROWSER_CHANNEL = CURRENT_EXEC_PATH = CURRENT_CDP_URL = None

                logger.info(f"webui.py stop completed (host={host}, port={port}).")

        try:
            fut = worker.submit_coro(_job_stop())
            fut.result()
            return ActionResp(
                status="ok",
                message="Stop completed: closed browser and terminated webui.py (handle+port ensured)."
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