import json
import os
import logging
from functools import partial
from typing import Any, Dict, Optional

import gradio as gr
from gradio.components import Component

from src.webui.webui_manager import WebuiManager
from src.utils import config

logger = logging.getLogger(__name__)

# =========================
# webui.json path and loading function
# =========================

# Fixed default path for webui.json (in the same directory as this file)
DEFAULT_WEBUI_JSON_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "webui.json"
)
# If the environment variable WEBUI_JSON_PATH is set, use it first
WEBUI_JSON_PATH = os.getenv("WEBUI_JSON_PATH", DEFAULT_WEBUI_JSON_PATH)


def load_llm_settings_from_webui_json() -> (Optional[str], Optional[str]):
    """
    Load llm_model_name and llm_base_url from webui.json.
    - Returns (model_name, base_url)
    - Returns corresponding None values if file doesn't exist or fields are missing
    """
    model_name = None
    base_url = None
    path = WEBUI_JSON_PATH
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            model_name = cfg.get("llm_model_name")
            base_url = cfg.get("llm_base_url")
            logger.info(
                f"[agent_settings_tab] Loaded LLM configuration from {path}: "
                f"llm_model_name={model_name}, llm_base_url={base_url}"
            )
        else:
            logger.info(f"[agent_settings_tab] webui.json not found: {path}")
    except Exception as e:
        logger.warning(f"[agent_settings_tab] Error reading {path}: {e}")

    return model_name, base_url


# =========================
# Other existing logic
# =========================

def update_model_dropdown(llm_provider):
    """
    Update the model name dropdown with predefined models for the selected provider.
    """
    # Use predefined models for the selected provider
    if llm_provider in config.model_names:
        return gr.Dropdown(
            choices=config.model_names[llm_provider],
            value=config.model_names[llm_provider][0],
            interactive=True
        )
    else:
        return gr.Dropdown(
            choices=[],
            value="",
            interactive=True,
            allow_custom_value=True
        )


async def update_mcp_server(mcp_file: str, webui_manager: WebuiManager):
    """
    Update the MCP server.
    """
    if hasattr(webui_manager, "bu_controller") and webui_manager.bu_controller:
        logger.warning("⚠️ Close controller because mcp file has changed!")
        await webui_manager.bu_controller.close_mcp_client()
        webui_manager.bu_controller = None

    if not mcp_file or not os.path.exists(mcp_file) or not mcp_file.endswith('.json'):
        logger.warning(f"{mcp_file} is not a valid MCP file.")
        return None, gr.update(visible=False)

    with open(mcp_file, 'r') as f:
        mcp_server = json.load(f)

    return json.dumps(mcp_server, indent=2), gr.update(visible=True)


def create_agent_settings_tab(webui_manager: WebuiManager):
    """
    Creates an agent settings tab.
    """
    input_components = set(webui_manager.get_components())
    tab_components = {}

    # === Load default LLM configuration from webui.json ===
    llm_model_default, llm_base_url_default = load_llm_settings_from_webui_json()

    # Set initial value and dropdown options for Provider
    initial_provider = "openai"
    initial_choices = config.model_names.get(initial_provider, [])
    # Model initial value priority: webui.json > first preset of provider > empty string
    initial_model_value = (
        llm_model_default if llm_model_default
        else (initial_choices[0] if initial_choices else "")
    )
    # Base URL initial value priority: webui.json > original default
    initial_base_url_value = llm_base_url_default or "https://api.openai-proxy.org/v1"

    with gr.Group():
        with gr.Column():
            override_system_prompt = gr.Textbox(
                label="Override system prompt", lines=4, interactive=True
            )
            extend_system_prompt = gr.Textbox(
                label="Extend system prompt", lines=4, interactive=True
            )

    with gr.Group():
        mcp_json_file = gr.File(
            label="MCP server json", interactive=True, file_types=[".json"]
        )
        mcp_server_config = gr.Textbox(
            label="MCP server", lines=6, interactive=True, visible=False
        )

    with gr.Group():
        with gr.Row():
            llm_provider = gr.Dropdown(
                choices=[provider for provider, model in config.model_names.items()],
                label="LLM Provider",
                value=initial_provider,
                info="Select LLM provider for LLM",
                interactive=True
            )
            llm_model_name = gr.Dropdown(
                label="LLM Model Name",
                # Provide choices based on initial_provider initially
                choices=initial_choices,
                value=initial_model_value,
                interactive=True,
                allow_custom_value=True,
                info="Select a model in the dropdown options or directly type a custom model name"
            )
        with gr.Row():
            llm_temperature = gr.Slider(
                minimum=0.0,
                maximum=2.0,
                value=0.6,
                step=0.1,
                label="LLM Temperature",
                info="Controls randomness in model outputs",
                interactive=True
            )

            use_vision = gr.Checkbox(
                label="Use Vision",
                value=False,
                info="Enable Vision(Input highlighted screenshot into LLM)",
                interactive=True
            )

            ollama_num_ctx = gr.Slider(
                minimum=2 **8,
                maximum=2** 16,
                value=16000,
                step=1,
                label="Ollama Context Length",
                info="Controls max context length model needs to handle (less = faster)",
                visible=False,
                interactive=True
            )

        with gr.Row():
            llm_base_url = gr.Textbox(
                label="Base URL",
                value=initial_base_url_value,
                info="API endpoint URL (if required)"
            )
            llm_api_key = gr.Textbox(
                label="API Key",
                type="password",
                value="",  # No longer preset any API key
                info="Your API key (leave blank to use .env)"
            )

    with gr.Group():
        with gr.Row():
            planner_llm_provider = gr.Dropdown(
                choices=[provider for provider, model in config.model_names.items()],
                label="Planner LLM Provider",
                info="Select LLM provider for LLM",
                value=None,
                interactive=True
            )
            planner_llm_model_name = gr.Dropdown(
                label="Planner LLM Model Name",
                interactive=True,
                allow_custom_value=True,
                info="Select a model in the dropdown options or directly type a custom model name"
            )
        with gr.Row():
            planner_llm_temperature = gr.Slider(
                minimum=0.0,
                maximum=2.0,
                value=0.6,
                step=0.1,
                label="Planner LLM Temperature",
                info="Controls randomness in model outputs",
                interactive=True
            )

            planner_use_vision = gr.Checkbox(
                label="Use Vision(Planner LLM)",
                value=False,
                info="Enable Vision(Input highlighted screenshot into LLM)",
                interactive=True
            )

            planner_ollama_num_ctx = gr.Slider(
                minimum=2 **8,
                maximum=2** 16,
                value=16000,
                step=1,
                label="Ollama Context Length",
                info="Controls max context length model needs to handle (less = faster)",
                visible=False,
                interactive=True
            )

        with gr.Row():
            planner_llm_base_url = gr.Textbox(
                label="Base URL",
                value="",
                info="API endpoint URL (if required)"
            )
            planner_llm_api_key = gr.Textbox(
                label="API Key",
                type="password",
                value="",
                info="Your API key (leave blank to use .env)"
            )

    with gr.Row():
        max_steps = gr.Slider(
            minimum=1,
            maximum=1000,
            value=100,
            step=1,
            label="Max Run Steps",
            info="Maximum number of steps the agent will take",
            interactive=True
        )
        max_actions = gr.Slider(
            minimum=1,
            maximum=100,
            value=10,
            step=1,
            label="Max Number of Actions",
            info="Maximum number of actions the agent will take per step",
            interactive=True
        )

    with gr.Row():
        max_input_tokens = gr.Number(
            label="Max Input Tokens",
            value=128000,
            precision=0,
            interactive=True
        )
        tool_calling_method = gr.Dropdown(
            label="Tool Calling Method",
            value="auto",
            interactive=True,
            allow_custom_value=True,
            choices=['function_calling', 'json_mode', 'raw', 'auto', 'tools', "None"],
            visible=True
        )

    tab_components.update(dict(
        override_system_prompt=override_system_prompt,
        extend_system_prompt=extend_system_prompt,
        llm_provider=llm_provider,
        llm_model_name=llm_model_name,
        llm_temperature=llm_temperature,
        use_vision=use_vision,
        ollama_num_ctx=ollama_num_ctx,
        llm_base_url=llm_base_url,
        llm_api_key=llm_api_key,
        planner_llm_provider=planner_llm_provider,
        planner_llm_model_name=planner_llm_model_name,
        planner_llm_temperature=planner_llm_temperature,
        planner_use_vision=planner_use_vision,
        planner_ollama_num_ctx=planner_ollama_num_ctx,
        planner_llm_base_url=planner_llm_base_url,
        planner_llm_api_key=planner_llm_api_key,
        max_steps=max_steps,
        max_actions=max_actions,
        max_input_tokens=max_input_tokens,
        tool_calling_method=tool_calling_method,
        mcp_json_file=mcp_json_file,
        mcp_server_config=mcp_server_config,
    ))
    webui_manager.add_components("agent_settings", tab_components)

    # === Interaction logic remains unchanged ===
    llm_provider.change(
        fn=lambda x: gr.update(visible=x == "ollama"),
        inputs=llm_provider,
        outputs=ollama_num_ctx
    )
    llm_provider.change(
        lambda provider: update_model_dropdown(provider),
        inputs=[llm_provider],
        outputs=[llm_model_name]
    )
    planner_llm_provider.change(
        fn=lambda x: gr.update(visible=x == "ollama"),
        inputs=[planner_llm_provider],
        outputs=[planner_ollama_num_ctx]
    )
    planner_llm_provider.change(
        lambda provider: update_model_dropdown(provider),
        inputs=[planner_llm_provider],
        outputs=[planner_llm_model_name]
    )

    async def update_wrapper(mcp_file):
        """Wrapper for handle_pause_resume."""
        update_dict = await update_mcp_server(mcp_file, webui_manager)
        yield update_dict

    mcp_json_file.change(
        update_wrapper,
        inputs=[mcp_json_file],
        outputs=[mcp_server_config, mcp_server_config]
    )