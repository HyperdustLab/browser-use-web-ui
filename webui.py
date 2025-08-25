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
# 日志：控制台 + WebSocket 广播
# -----------------------------

# 强制无缓冲，保证实时输出（http_run.py 也会设，这里再兜底）
os.environ.setdefault("PYTHONUNBUFFERED", "1")

# 统一日志格式
LOG_FMT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_console_handler = logging.StreamHandler(sys.__stdout__)
_console_handler.setFormatter(logging.Formatter(LOG_FMT))

logger = logging.getLogger("webui-main")
logger.setLevel(logging.DEBUG)
logger.handlers.clear()
logger.addHandler(_console_handler)
logger.propagate = False

# 同时提高常见依赖的日志等级，便于排查底层服务
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
# WebSocket 广播（保留你的设计）
# -----------------------------

websocket_clients = set()
websocket_loop = None  # ✅ 全局事件循环引用

def debug_plain(msg: str):
    """直接写到父进程控制台；不走 logging，避免死循环。"""
    try:
        sys.__stdout__.write(f"{msg}\n")
        sys.__stdout__.flush()
    except Exception:
        pass

# ✅ WebSocket 客户端处理器
async def websocket_handler(websocket):
    websocket_clients.add(websocket)
    debug_plain(f"[WS] 客户端已连接，当前连接数: {len(websocket_clients)}")
    try:
        async for _ in websocket:
            pass
    except Exception as e:
        debug_plain(f"[WS] 接收消息异常: {e}")
    finally:
        websocket_clients.remove(websocket)
        debug_plain(f"[WS] 客户端断开连接，剩余连接数: {len(websocket_clients)}")

# ✅ WebSocket 服务主循环
async def websocket_server():
    global websocket_loop
    websocket_loop = asyncio.get_running_loop()
    debug_plain("[WS] 启动 WebSocket 服务：ws://0.0.0.0:8765")
    async with websockets.serve(websocket_handler, "0.0.0.0", 8765, ping_interval=None):
        await asyncio.Future()

def start_websocket_server():
    asyncio.run(websocket_server())

# ✅ 广播日志消息到 WS，同时也会写控制台（由 TeeLoggerStream 保证）
def broadcast_log_message(message):
    if not websocket_clients:
        # 控制台可见，方便确认“没人连上”
        debug_plain(f"[WS] 无客户端连接，跳过广播：{message}")
        return

    debug_plain(f"[WS] 广播消息：{message}，连接数: {len(websocket_clients)}")
    send_tasks = []
    for ws in websocket_clients.copy():
        try:
            send_tasks.append(ws.send(message))
        except Exception as e:
            debug_plain(f"[WS] 过滤 ws 失败: {e}")

    if send_tasks and websocket_loop:
        try:
            future = asyncio.run_coroutine_threadsafe(
                asyncio.gather(*send_tasks, return_exceptions=True),
                websocket_loop
            )
            future.result()
        except Exception as e:
            debug_plain(f"[WS] 广播 handler 异常: {e}")

# ✅ 标准输出流拦截（用于捕获 print 和第三方输出）
class TeeLoggerStream:
    def __init__(self, original_stream):
        self.original_stream = original_stream
        self._buffer = ""

    def write(self, message):
        # 1) 原样写回控制台
        try:
            self.original_stream.write(message)
            self.original_stream.flush()
        except Exception:
            pass

        # 2) 行缓冲，逐行广播
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
        pass  # 无文件写入

# ✅ logging handler → 广播（不影响控制台 handler）
class BroadcastingHandler(logging.Handler):
    def emit(self, record):
        try:
            msg = self.format(record)
            broadcast_log_message(msg)
        except Exception as e:
            debug_plain(f"[WS] 广播 handler 异常: {e}")

# ✅ 将广播 handler 附加到所有 logger（保留控制台 handler）
def attach_broadcast_to_all_loggers():
    b_handler = BroadcastingHandler()
    b_handler.setFormatter(logging.Formatter(LOG_FMT))

    # 根 logger
    root_logger = logging.getLogger()
    if not any(isinstance(h, BroadcastingHandler) for h in root_logger.handlers):
        root_logger.addHandler(b_handler)

    # 已存在的所有 logger
    for name in list(logging.root.manager.loggerDict.keys()):
        lg = logging.getLogger(name)
        if not any(isinstance(h, BroadcastingHandler) for h in lg.handlers):
            lg.addHandler(b_handler)
        # 不修改它们的 console handler；等级前面已经统一提高

# -----------------------------
# 启动 Gradio WebUI
# -----------------------------

def main():
    parser = argparse.ArgumentParser(description="Gradio WebUI for Browser Agent")
    parser.add_argument("--ip", type=str, default="127.0.0.1", help="IP address to bind to")
    parser.add_argument("--port", type=int, default=7788, help="Port to listen on")
    parser.add_argument("--theme", type=str, default="Ocean", choices=theme_map.keys(), help="Theme to use for the UI")
    args = parser.parse_args()

    # 启动参数回显
    logger.info("启动参数：ip=%s, port=%s, theme=%s", args.ip, args.port, args.theme)
    logger.debug("环境变量关键信息：PYTHONUNBUFFERED=%s", os.environ.get("PYTHONUNBUFFERED"))

    try:
        logger.info("创建 Gradio 界面 ...")
        demo = create_ui(theme_name=args.theme)
        logger.info("启动 Gradio（%s:%s） ...", args.ip, args.port)
        # 展示错误细节，有助排查
        demo.queue().launch(server_name=args.ip, server_port=args.port, show_error=True)
    except Exception:
        logger.exception("Gradio 启动失败")
        raise

# ✅ 主程序入口
if __name__ == '__main__':
    # 启动 WS 日志广播线程
    debug_plain("[MAIN] 启动 WebSocket 线程 ...")
    threading.Thread(target=start_websocket_server, daemon=True).start()

    # 将广播 handler 附加到所有 logger，让 logging 日志也能被 WS 收到
    attach_broadcast_to_all_loggers()

    # 把 stdout/stderr 的“行”广播出去，同时保留在控制台
    tee_stdout = TeeLoggerStream(sys.stdout)
    tee_stderr = TeeLoggerStream(sys.stderr)

    # 同时把 print()/第三方输出 → 控制台 & WS
    with contextlib.redirect_stdout(tee_stdout), contextlib.redirect_stderr(tee_stderr):
        main()
