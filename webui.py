from dotenv import load_dotenv
load_dotenv()

import argparse
import sys
import os
import contextlib
import logging
import asyncio
import threading
import websockets

from src.webui.interface import theme_map, create_ui

# -----------------------------
# Logging: Console + WebSocket Broadcast
# -----------------------------

# Force unbuffered to ensure real-time output (http_run.py also sets this, here as backup)
os.environ.setdefault("PYTHONUNBUFFERED", "1")

# Unified log format
LOG_FMT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_console_handler = logging.StreamHandler(sys.__stdout__)
_console_handler.setFormatter(logging.Formatter(LOG_FMT))

logger = logging.getLogger("webui-main")
logger.setLevel(logging.DEBUG)
logger.handlers.clear()
logger.addHandler(_console_handler)
logger.propagate = False

# Also increase log level for common dependencies to facilitate debugging underlying services
for _name, _level in [
    ("uvicorn", logging.DEBUG),
    ("uvicorn.error", logging.DEBUG),
    ("uvicorn.access", logging.DEBUG),
    ("gradio", logging.DEBUG),
    ("asyncio", logging.INFO),
]:
    _lg = logging.getLogger(_name)
    _lg.setLevel(_level)
    if not any(isinstance(h, logging.StreamHandler) for h in _lg.handlers):
        _lg.addHandler(_console_handler)
    _lg.propagate = False

# -----------------------------
# WebSocket Broadcast (preserve your design)
# -----------------------------

websocket_clients = set()
websocket_loop = None  # ✅ Global event loop reference

def debug_plain(msg: str):
    """Write directly to parent process console; bypass logging to avoid infinite loops."""
    try:
        sys.__stdout__.write(f"{msg}\n")
        sys.__stdout__.flush()
    except Exception:
        pass

# ✅ WebSocket client handler
async def websocket_handler(websocket):
    websocket_clients.add(websocket)
    debug_plain(f"[WS] Client connected, current connections: {len(websocket_clients)}")
    try:
        async for _ in websocket:
            pass
    except Exception as e:
        debug_plain(f"[WS] Exception receiving message: {e}")
    finally:
        websocket_clients.remove(websocket)
        debug_plain(f"[WS] Client disconnected, remaining connections: {len(websocket_clients)}")

# ✅ WebSocket service main loop
async def websocket_server():
    global websocket_loop
    websocket_loop = asyncio.get_running_loop()
    debug_plain("[WS] Starting WebSocket service: ws://0.0.0.0:8765")
    async with websockets.serve(websocket_handler, "0.0.0.0", 8765, ping_interval=None):
        await asyncio.Future()

def start_websocket_server():
    asyncio.run(websocket_server())

# ✅ Broadcast log messages to WS, also write to console (guaranteed by TeeLoggerStream)
def broadcast_log_message(message):
    if not websocket_clients:
        # Console visible, convenient to confirm "no one connected"
        debug_plain(f"[WS] No client connections, skipping broadcast: {message}")
        return

    debug_plain(f"[WS] Broadcasting message: {message}, connections: {len(websocket_clients)}")
    send_tasks = []
    for ws in websocket_clients.copy():
        try:
            send_tasks.append(ws.send(message))
        except Exception as e:
            debug_plain(f"[WS] Failed to filter ws: {e}")

    if send_tasks and websocket_loop:
        try:
            future = asyncio.run_coroutine_threadsafe(
                asyncio.gather(*send_tasks, return_exceptions=True),
                websocket_loop
            )
            future.result()
        except Exception as e:
            debug_plain(f"[WS] Broadcast handler exception: {e}")

# ✅ Standard output stream interception (for capturing print and third-party output)
class TeeLoggerStream:
    def __init__(self, original_stream):
        self.original_stream = original_stream
        self._buffer = ""

    def write(self, message):
        # 1) Write back to console as is
        try:
            self.original_stream.write(message)
            self.original_stream.flush()
        except Exception:
            pass

        # 2) Line buffering, broadcast line by line
        self._buffer += message
        if "\n" in self._buffer:
            lines = self._buffer.split("\n")
            for line in lines[:-1]:
                line = line.strip()
                if line:
                    try:
                        broadcast_log_message(line)
                    except Exception:
                        pass
            self._buffer = lines[-1]

    def flush(self):
        try:
            self.original_stream.flush()
        except Exception:
            pass

    def isatty(self):
        return getattr(self.original_stream, "isatty", lambda: False)()

    def close(self):
        pass  # No file writing

# ✅ logging handler → broadcast (doesn't affect console handler)
class BroadcastLoggingHandler(logging.Handler):
    def emit(self, record):
        try:
            msg = self.format(record)
            broadcast_log_message(msg)
        except Exception as e:
            debug_plain(f"[WS] Broadcast handler exception: {e}")

# ✅ Attach broadcast handler to all loggers (preserve console handler)
def attach_broadcast_handler_to_all_loggers():
    broadcast_handler = BroadcastLoggingHandler()
    broadcast_handler.setFormatter(logging.Formatter(LOG_FMT))

    # Root logger
    root_logger = logging.getLogger()
    root_logger.addHandler(broadcast_handler)

    # All existing loggers
    for name in logging.root.manager.loggerDict:
        logger_obj = logging.getLogger(name)
        # Don't modify their console handlers; levels already unified above
        logger_obj.addHandler(broadcast_handler)

# -----------------------------
# Start Gradio WebUI
# -----------------------------

def main():
    parser = argparse.ArgumentParser(description="WebUI for Browser Use Agent")
    parser.add_argument("--ip", default="127.0.0.1", help="Server IP")
    parser.add_argument("--port", type=int, default=7788, help="Server port")
    parser.add_argument("--theme", default="default", help="Theme name")
    args = parser.parse_args()

    # Launch argument echo
    logger.info("Launching with args: ip=%s, port=%s, theme=%s", args.ip, args.port, args.theme)
    logger.debug("Environment variable key info: PYTHONUNBUFFERED=%s", os.environ.get("PYTHONUNBUFFERED"))

    try:
        logger.info("Creating Gradio interface ...")
        demo = create_ui(theme_name=args.theme)
        logger.info("Starting Gradio (%s:%s) ...", args.ip, args.port)
        # Show error details to help debugging
        demo.queue().launch(server_name=args.ip, server_port=args.port, show_error=True)
    except Exception:
        logger.exception("Gradio startup failed")
        raise

# ✅ Main program entry
if __name__ == '__main__':
    # Start WS log broadcast thread
    debug_plain("[MAIN] Starting WebSocket thread ...")
    threading.Thread(target=start_websocket_server, daemon=True).start()

    # Attach broadcast handler to all loggers so logging logs can also be received by WS
    attach_broadcast_handler_to_all_loggers()

    # Broadcast stdout/stderr "lines" while preserving them in console
    sys.stdout = TeeLoggerStream(sys.__stdout__)
    sys.stderr = TeeLoggerStream(sys.__stderr__)

    main()
