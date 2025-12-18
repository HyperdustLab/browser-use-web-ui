from dotenv import load_dotenv
load_dotenv()

import argparse
import sys
import contextlib
import logging
import asyncio
import threading
import websockets
import os  # ✅ 新增：读取环境变量
import signal  # ✅ 新增：接管 Ctrl+C
from src.webui.interface import theme_map, create_ui


websocket_clients = set()
websocket_loop = None  # ✅ Global event loop reference

# ✅ Safe debug output
def debug(msg):
    sys.__stdout__.write(f"{msg}\n")
    sys.__stdout__.flush()

# ✅ WebSocket client handler
async def websocket_handler(websocket):
    websocket_clients.add(websocket)
    debug(f"[WS] Client connected, current connections: {len(websocket_clients)}")
    try:
        async for _ in websocket:
            pass
    except Exception as e:
        debug(f"[WS] Message reception error: {e}")
    finally:
        websocket_clients.remove(websocket)
        debug(f"[WS] Client disconnected, remaining connections: {len(websocket_clients)}")

# ✅ WebSocket server main loop（端口从环境变量读取）
async def websocket_server():
    global websocket_loop
    websocket_loop = asyncio.get_running_loop()

    # ✅ 关键修改点：从环境变量中读取 WS 端口，默认 8765
    ws_port = int(os.environ.get("WEBUI_WS_PORT", "8765"))

    debug(f"[WS] Starting WebSocket service: ws://0.0.0.0:{ws_port}")
    async with websockets.serve(websocket_handler, "0.0.0.0", ws_port, ping_interval=None):
        await asyncio.Future()

def start_websocket_server():
    asyncio.run(websocket_server())

# ✅ Broadcast log messages
def broadcast_log_message(message):
    if not websocket_clients:
        debug(f"[WS] No clients connected, skipping broadcast: {message}")
        return

    debug(f"[WS] Broadcasting message: {message}, connections: {len(websocket_clients)}")
    send_tasks = []
    for ws in websocket_clients.copy():
        try:
            send_tasks.append(ws.send(message))
        except Exception as e:
            debug(f"[WS] Failed to filter ws: {e}")

    if send_tasks and websocket_loop:
        try:
            async def _send_all():
                return await asyncio.gather(*send_tasks, return_exceptions=True)

            future = asyncio.run_coroutine_threadsafe(_send_all(), websocket_loop)
            future.result()
        except Exception as e:
            debug(f"[WS] Broadcast handler error: {e}")


# ✅ Standard output stream interception (for capturing print and third-party output)
class TeeLoggerStream:
    def __init__(self, original_stream):
        self.original_stream = original_stream
        self._buffer = ""

    def write(self, message):
        self.original_stream.write(message)
        self.original_stream.flush()
        self._buffer += message
        if "\n" in self._buffer:
            lines = self._buffer.split("\n")
            for line in lines[:-1]:
                if line.strip():
                    broadcast_log_message(line.strip())
            self._buffer = lines[-1]

    def flush(self):
        self.original_stream.flush()

    def isatty(self):
        return self.original_stream.isatty()

    def close(self):
        pass  # No file writing

# ✅ logging handler → broadcast
class BroadcastingHandler(logging.Handler):
    def emit(self, record):
        try:
            msg = self.format(record)
            broadcast_log_message(msg)
        except Exception as e:
            debug(f"[WS] Broadcast handler error: {e}")

# ✅ Clear all existing handlers and attach broadcast handler (prevent duplication)
def attach_handler_to_all_loggers(*handlers):
    # Clean up existing handlers for all loggers
    for name in logging.root.manager.loggerDict:
        logging.getLogger(name).handlers.clear()
    logging.getLogger().handlers.clear()

    # Add handlers and disable upward propagation
    loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
    loggers.append(logging.getLogger())
    for logger in loggers:
        for handler in handlers:
            if handler not in logger.handlers:
                logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False  # ✅ Disable propagation to prevent duplicate broadcasts

# ✅ Start Gradio WebUI
def main():
    parser = argparse.ArgumentParser(description="Gradio WebUI for Browser Agent")
    parser.add_argument("--ip", type=str, default="127.0.0.1", help="IP address to bind to")
    parser.add_argument("--port", type=int, default=7788, help="Port to listen on")
    parser.add_argument("--theme", type=str, default="Ocean", choices=theme_map.keys(), help="Theme to use for the UI")
    args = parser.parse_args()

    debug(f"[INFO] Starting WebUI service: http://{args.ip}:{args.port}")
    demo = create_ui(theme_name=args.theme)
    demo.queue().launch(server_name=args.ip, server_port=args.port)

# ✅ Main program entry
if __name__ == '__main__':
    # ✅ 方案 A：接管 Ctrl+C，直接强退，避免卡在 “Press [Enter] to resume...”
    def _hard_exit(signum, frame):
        os._exit(0)

    signal.signal(signal.SIGINT, _hard_exit)  # Ctrl+C
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, _hard_exit)  # Windows: Ctrl+Break

    debug("[MAIN] Starting WebSocket thread...")
    threading.Thread(target=start_websocket_server, daemon=True).start()

    tee_stdout = TeeLoggerStream(sys.stdout)
    tee_stderr = TeeLoggerStream(sys.stderr)

    broadcast_handler = BroadcastingHandler()
    broadcast_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s'))

    attach_handler_to_all_loggers(broadcast_handler)

    with contextlib.redirect_stdout(tee_stdout), contextlib.redirect_stderr(tee_stderr):
        main()
