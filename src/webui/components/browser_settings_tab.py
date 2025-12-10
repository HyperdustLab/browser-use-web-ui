import os
import json
from distutils.util import strtobool
import gradio as gr
import logging
from gradio.components import Component  # Keep original import
from typing import Optional

from src.webui.webui_manager import WebuiManager
from src.utils import config

logger = logging.getLogger(__name__)

# Fixed webui.json default path (under the directory where this file is located)
DEFAULT_WEBUI_JSON_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "webui.json"
)

# If environment variable WEBUI_JSON_PATH is set, use it first, otherwise use default path
WEBUI_JSON_PATH = os.getenv("WEBUI_JSON_PATH", DEFAULT_WEBUI_JSON_PATH)

def load_cdp_url_from_webui_json() -> Optional[str]:
    """
    Dynamically read cdp_URP from webui.json as CDP URL.
    Fallback: environment variable BROWSER_CDP; if still none, return None.
    """
    path = WEBUI_JSON_PATH
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            # Your JSON format is { "cdp_URP": "http://43.159.42.14" }
            val = cfg.get("cdp_URP")
            if val:
                logger.info(f"[browser_settings_tab] Using cdp_URP from {path}: {val}")
                return val
            else:
                logger.info(f"[browser_settings_tab] {path} exists but cdp_URP field not found")
        else:
            logger.info(f"[browser_settings_tab] webui.json not found: {path}")
    except Exception as e:
        logger.warning(f"[browser_settings_tab] Error reading {path}: {e}")

    # Fallback to environment variable
    env_val = os.getenv("BROWSER_CDP")
    if env_val:
        logger.info(f"[browser_settings_tab] Using environment variable BROWSER_CDP: {env_val}")
        return env_val

    logger.info("[browser_settings_tab] No CDP URL obtained (returning None)")
    return None


async def close_browser(webui_manager: WebuiManager):
    """
    Close browser
    """
    if webui_manager.bu_current_task and not webui_manager.bu_current_task.done():
        webui_manager.bu_current_task.cancel()
        webui_manager.bu_current_task = None

    if webui_manager.bu_browser_context:
        logger.info("⚠️ Closing browser context when changing browser config.")
        await webui_manager.bu_browser_context.close()
        webui_manager.bu_browser_context = None

    if webui_manager.bu_browser:
        logger.info("⚠️ Closing browser when changing browser config.")
        await webui_manager.bu_browser.close()
        webui_manager.bu_browser = None


def create_browser_settings_tab(webui_manager: WebuiManager):
    """
    Creates a browser settings tab.
    """
    input_components = set(webui_manager.get_components())
    tab_components = {}

    with gr.Group():
        with gr.Row():
            browser_binary_path = gr.Textbox(
                label="Browser Binary Path",
                lines=1,
                interactive=True,
                placeholder="e.g. '/Applications/Google\\ Chrome.app/Contents/MacOS/Google\\ Chrome'"
            )
            browser_user_data_dir = gr.Textbox(
                label="Browser User Data Dir",
                lines=1,
                interactive=True,
                placeholder="Leave it empty if you use your default user data",
            )
    with gr.Group():
        with gr.Row():
            use_own_browser = gr.Checkbox(
                label="Use Own Browser",
                value=bool(strtobool(os.getenv("USE_OWN_BROWSER", "false"))),
                info="Use your existing browser instance",
                interactive=True
            )
            keep_browser_open = gr.Checkbox(
                label="Keep Browser Open",
                value=bool(strtobool(os.getenv("KEEP_BROWSER_OPEN", "true"))),
                info="Keep Browser Open between Tasks",
                interactive=True
            )
            headless = gr.Checkbox(
                label="Headless Mode",
                value=False,
                info="Run browser without GUI",
                interactive=True
            )
            disable_security = gr.Checkbox(
                label="Disable Security",
                value=False,
                info="Disable browser security",
                interactive=True
            )

    with gr.Group():
        with gr.Row():
            window_w = gr.Number(
                label="Window Width",
                value=1280,
                info="Browser window width",
                interactive=True
            )
            window_h = gr.Number(
                label="Window Height",
                value=1100,
                info="Browser window height",
                interactive=True
            )
    with gr.Group():
        with gr.Row():
            # ✅ Here dynamically read webui.json when rendering UI (not when module imports)
            cdp_url = gr.Textbox(
                label="CDP URL",
                value=load_cdp_url_from_webui_json(),
                info="CDP URL for browser remote debugging",
                interactive=True,
            )
            wss_url = gr.Textbox(
                label="WSS URL",
                info="WSS URL for browser remote debugging",
                interactive=True,
            )
    with gr.Group():
        with gr.Row():
            save_recording_path = gr.Textbox(
                label="Recording Path",
                placeholder="e.g. ./tmp/record_videos",
                info="Path to save browser recordings",
                interactive=True,
            )

            save_trace_path = gr.Textbox(
                label="Trace Path",
                placeholder="e.g. ./tmp/traces",
                info="Path to save Agent traces",
                interactive=True,
            )

        with gr.Row():
            save_agent_history_path = gr.Textbox(
                label="Agent History Save Path",
                value="./tmp/agent_history",
                info="Specify the directory where agent history should be saved.",
                interactive=True,
            )
            save_download_path = gr.Textbox(
                label="Save Directory for browser downloads",
                value="./tmp/downloads",
                info="Specify the directory where downloaded files should be saved.",
                interactive=True,
            )
    tab_components.update(
        dict(
            browser_binary_path=browser_binary_path,
            browser_user_data_dir=browser_user_data_dir,
            use_own_browser=use_own_browser,
            keep_browser_open=keep_browser_open,
            headless=headless,
            disable_security=disable_security,
            save_recording_path=save_recording_path,
            save_trace_path=save_trace_path,
            save_agent_history_path=save_agent_history_path,
            save_download_path=save_download_path,
            cdp_url=cdp_url,
            wss_url=wss_url,
            window_h=window_h,
            window_w=window_w,
        )
    )
    webui_manager.add_components("browser_settings", tab_components)

    async def close_wrapper():
        """Wrapper for handle_clear."""
        await close_browser(webui_manager)

    headless.change(close_wrapper)
    keep_browser_open.change(close_wrapper)
    disable_security.change(close_wrapper)
    use_own_browser.change(close_wrapper)
