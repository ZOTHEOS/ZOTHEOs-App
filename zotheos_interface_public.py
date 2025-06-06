import gradio as gr
import asyncio
import logging
import os
import sys
import time
import random
from typing import Optional, Dict, Any, List, Union, Tuple
import html
import json
import tempfile

# --- Constants ---
APP_TITLE = "ZOTHEOS - Ethical Fusion AI" # This can be imported by main_web.py if needed
FAVICON_FILENAME = "favicon_blackBW.ico"
LOGO_FILENAME = "zotheos_logo.png"

# --- Logger Setup ---
def init_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    logger_instance = logging.getLogger("ZOTHEOS_Interface")
    logger_instance.propagate = False
    if not logger_instance.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger_instance.addHandler(handler)
    return logger_instance
logger = init_logger() # logger is defined at module level

# --- Asset Path Helper (Corrected for PyInstaller) ---
def get_asset_path(filename: str) -> Optional[str]:
    if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
        bundle_dir = sys._MEIPASS # type: ignore
        possible_path = os.path.join(bundle_dir, 'assets', filename)
        if os.path.exists(possible_path):
            logger.info(f"Asset found in PyInstaller bundle: '{filename}' -> '{possible_path}'")
            return possible_path
        possible_path_root = os.path.join(bundle_dir, filename)
        if os.path.exists(possible_path_root):
            logger.info(f"Asset found at PyInstaller bundle root: '{filename}' -> '{possible_path_root}'")
            return possible_path_root
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        path_in_assets = os.path.join(script_dir, "assets", filename)
        if os.path.exists(path_in_assets):
            logger.info(f"Asset found in local ./assets: '{filename}' -> '{path_in_assets}'")
            return path_in_assets
        path_in_root = os.path.join(script_dir, filename)
        if os.path.exists(path_in_root):
            logger.info(f"Asset found in local script dir: '{filename}' -> '{path_in_root}'")
            return path_in_root
    logger.warning(f"Asset NOT FOUND: '{filename}'.")
    return None

# --- Define Asset Paths at Module Level for Import by main_web.py ---
logo_path_verified = get_asset_path(LOGO_FILENAME)
favicon_path_verified = get_asset_path(FAVICON_FILENAME)

# --- Core Logic Import Attempt ---
MainFusionPublic = None
initialization_error: Optional[Exception] = None
DEFAULT_INFERENCE_PRESET_INTERFACE = "balanced"
get_user_tier_fallback = lambda token: "error (auth module failed)"
get_user_tier = get_user_tier_fallback

try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = script_dir
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
        logger.info(f"ADDED project root directory '{project_root}' to sys.path for module resolution.")
    from modules.main_fusion_public import MainFusionPublic # type: ignore
    try:
        from modules.config_settings_public import DEFAULT_INFERENCE_PRESET as CONFIG_DEFAULT_PRESET # type: ignore
        DEFAULT_INFERENCE_PRESET_INTERFACE = CONFIG_DEFAULT_PRESET
        logger.info(f"Imported DEFAULT_INFERENCE_PRESET ('{DEFAULT_INFERENCE_PRESET_INTERFACE}') from config.")
    except ImportError:
        logger.warning("Could not import DEFAULT_INFERENCE_PRESET. Using interface default: 'balanced'")
    from modules.user_auth import get_user_tier as imported_get_user_tier # type: ignore
    get_user_tier = imported_get_user_tier
    logger.info("✅ Successfully imported MainFusionPublic and get_user_tier from modules.user_auth.")
except Exception as e:
    logger.error(f"❌ Error importing modules (MainFusionPublic or user_auth): {e}", exc_info=True)
    initialization_error = e
    if 'imported_get_user_tier' not in globals():
        get_user_tier = get_user_tier_fallback
        logger.info("Using fallback 'get_user_tier' due to specific import error for get_user_tier.")

# --- Initialize AI System ---
ai_system: Optional[MainFusionPublic] = None
if 'MainFusionPublic' in globals() and MainFusionPublic is not None and initialization_error is None:
    try:
        logger.info("Initializing ZOTHEOS AI System...")
        ai_system = MainFusionPublic() # type: ignore
        if not hasattr(ai_system, 'process_query_with_fusion'):
            missing_method_error = AttributeError("AI System lacks 'process_query_with_fusion' method.")
            logger.error(f"❌ AI System Config Error: {missing_method_error}")
            initialization_error = missing_method_error; ai_system = None
        else: logger.info("✅ ZOTHEOS AI System Initialized.")
    except Exception as e_init:
        initialization_error = e_init; logger.error(f"❌ AI System Init Failed: {e_init}", exc_info=True); ai_system = None
elif initialization_error is None and ('MainFusionPublic' not in globals() or MainFusionPublic is None) :
    initialization_error = ModuleNotFoundError("MainFusionPublic module could not be imported or was not defined.")
    logger.error(f"❌ {initialization_error}")

# --- TIER_FEATURES Dictionary ---
TIER_FEATURES = {
    "free": {"display_name": "Free Tier", "memory_enabled": False, "export_enabled": False, "settings_access": False, "emoji_map": {"memory": "❌", "export": "❌", "settings": "❌"}},
    "starter": {"display_name": "Starter Tier", "memory_enabled": True, "export_enabled": False, "settings_access": False, "emoji_map": {"memory": "✅", "export": "❌", "settings": "❌"}},
    "pro": {"display_name": "Pro Tier", "memory_enabled": True, "export_enabled": True, "settings_access": True, "emoji_map": {"memory": "✅", "export": "✅", "settings": "✅"}},
    "error (auth module failed)": {"display_name": "Error Resolving Tier", "memory_enabled": False, "export_enabled": False, "settings_access": False, "emoji_map": {"memory": "⚠️", "export": "⚠️", "settings": "⚠️"}}
}

zotheos_base_css = """
@import url('https://fonts.googleapis.com/css2?family=Lato:wght@400;700&family=Staatliches&family=Inconsolata:wght@400;700&display=swap');

:root {
    --background-main: #000000;
    --background-card: #0f0f0f;
    --input-bg: #121212;
    --text-color-light: #ffffff;
    --text-color-medium: #b3b3b3;
    --text-color-heading: #ffffff;
    --border-color: #2a2a2a;
    --input-border: var(--border-color);
    --input-focus-border: #888888;
    --accent-primary: #dcdcdc;
    --accent-secondary: #444444;

    --font-body: 'Lato', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    --font-heading: 'Staatliches', cursive;
    --font-mono: 'Inconsolata', monospace;

    --button-primary-bg: var(--accent-primary);
    --button-primary-text: #000000;
    --button-primary-hover-bg: #c0c0c0;
    --button-primary-focus-ring: #b0b0b0;

    --button-secondary-bg: var(--accent-secondary);
    --button-secondary-text: var(--text-color-light);
    --button-secondary-hover-bg: #555555;
    --button-secondary-focus-ring: #666666;
}

/* Base dark mode and pixelated rendering for a sharp, retro feel */
html, body {
    color-scheme: dark;
    image-rendering: pixelated;
}

/* 🧼 CANVAS CLEANUP (Gradio branding block removal) */
canvas {
    display: none !important;
}

/* 💥 FULL GRADIO FOOTER REMOVE */
footer {
    display: none !important;
}

body, html {
    background-color: var(--background-main) !important;
    margin: 0;
    padding: 0;
    font-family: var(--font-body);
    color: var(--text-color-light);
    line-height: 1.6;
    overflow-x: hidden;
}

body, .gradio-container {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: flex-start;
}

/* 💡 LIMIT WIDTH FOR MOBILE AND KEEP CENTERED */
.container, .gradio-container, .main, body {
    max-width: 720px !important;
    margin: 0 auto !important;
}

.gradio-container {
    background-color: transparent !important;
    box-shadow: none !important;
    padding: 0 !important;
}

.centered-container {
    max-width: 1080px;
    padding: 20px 15px;
    display: flex;
    flex-direction: column;
    align-items: center;
}

h1, h2, h3 {
    font-family: var(--font-heading);
    font-size: 1.5rem;
    text-transform: uppercase;
    color: var(--text-color-heading);
    letter-spacing: 1px;
}

#header_column {
    width: 100%;
    display: flex;
    flex-direction: column;
    align-items: center;
    text-align: center;
    margin-bottom: 1rem;
    padding-bottom: 10px;
}

#header_logo img {
    max-height: 100px;
    margin: 0.5rem auto;
    background-color: #000000;
    padding: 10px;
    border-radius: 12px;
}

#header_subtitle p {
    font-family: var(--font-body) !important;
    font-size: 1.1rem;
    color: var(--text-color-medium);
    text-align: center;
    margin: 0.25rem 0 1rem 0;
}

.interface-section,
#status_indicator,
#tools_accordion {
    width: 100% !important;
    max-width: 100% !important;
    box-sizing: border-box !important;
}

#query_input textarea {
    min-height: 160px !important;
    font-size: 1.05rem !important;
    background-color: var(--input-bg) !important;
    color: var(--text-color-light) !important;
    border: 1px solid var(--input-border) !important;
    border-radius: 10px !important;
    padding: 14px !important;
    line-height: 1.6 !important;
    font-family: var(--font-body) !important;
}

#fusion_output h1, #fusion_output h2, #fusion_output h3 {
    font-family: var(--font-heading) !important;
    font-size: clamp(1.3rem, 2vw, 1.8rem) !important;
    color: var(--accent-primary);
    border-bottom: 1px solid var(--border-color);
    padding-bottom: 0.3em;
    margin-top: 1em;
    margin-bottom: 0.5em;
}

#synthesized_summary_output h2 {
    font-family: var(--font-heading) !important;
    color: var(--accent-primary) !important;
    font-size: 1.5rem !important;
    border-bottom: 1px solid var(--border-color);
    padding-bottom: 0.3em;
    margin-bottom: 0.8em;
}

#synthesized_summary_output h3 {
    font-family: var(--font-heading) !important;
    color: var(--text-color-heading) !important;
    font-size: 1.1rem !important;
    margin-top: 0.8em;
    margin-bottom: 0.3em;
}

#memory_log textarea {
    font-family: var(--font-mono) !important;
    font-size: 0.8rem !important;
    line-height: 1.5 !important;
    background-color: #101015 !important;
    color: var(--text-color-medium) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 8px !important;
    padding: 10px !important;
}

/* Buttons */
.gradio-button.primary, button#submit_button.gradio-button {
    background: var(--button-primary-bg) !important;
    color: var(--button-primary-text) !important;
    border: 1px solid var(--button-primary-bg) !important;
    box-shadow: none !important;
}
.gradio-button.primary:hover, button#submit_button.gradio-button:hover {
    background: var(--button-primary-hover-bg) !important;
    border-color: var(--button-primary-hover-bg) !important;
}
.gradio-button.primary:focus, button#submit_button.gradio-button:focus {
    box-shadow: 0 0 0 2px var(--background-main),
                0 0 0 4px var(--button-primary-focus-ring) !important;
}

.gradio-button.secondary, button#clear_button.gradio-button {
    background: var(--button-secondary-bg) !important;
    color: var(--button-secondary-text) !important;
    border: 1px solid var(--button-secondary-bg) !important;
    box-shadow: none !important;
}
.gradio-button.secondary:hover, button#clear_button.gradio-button:hover {
    background: var(--button-secondary-hover-bg) !important;
    border-color: var(--button-secondary-hover-bg) !important;
}
.gradio-button.secondary:focus, button#clear_button.gradio-button:focus {
    box-shadow: 0 0 0 2px var(--background-main),
                0 0 0 4px var(--button-secondary-focus-ring) !important;
}

/* Pro Button Styling */
#pro_features_row .gradio-button {
    font-size: 0.85rem !important;
    padding: 8px 16px !important;
    flex-grow: 0 !important;
    background-color: var(--accent-secondary) !important;
    color: var(--text-color-medium) !important;
    opacity: 0.7;
    border: 1px solid var(--accent-secondary) !important;
}
#pro_features_row .gradio-button.interactive_button_enabled {
    background-color: var(--button-secondary-bg) !important;
    color: var(--button-secondary-text) !important;
    opacity: 1.0;
    border: 1px solid var(--button-secondary-bg) !important;
}
#pro_features_row .gradio-button.interactive_button_enabled:hover {
    background-color: var(--button-secondary-hover-bg) !important;
    border-color: var(--button-secondary-hover-bg) !important;
}
#pro_features_row .gradio-button:disabled,
#pro_features_row .gradio-button[disabled] {
    opacity: 0.4 !important;
    cursor: not-allowed !important;
    background-color: #202020 !important;
    color: #555 !important;
    border: 1px solid #202020 !important;
}

/* Attribution footer */
#footer_attribution {
    font-size: 0.75rem;
    color: var(--text-color-medium);
    text-align: center;
    padding-top: 1rem;
    margin-top: 2rem;
    opacity: 0.8;
    line-height: 1.5;
}

/* ✅ FORCE FULL RESPONSIVE LAYOUT ON MOBILE */
html, body {
    max-width: 100vw !important;
    overflow-x: hidden !important;
}

.gradio-container,
.container,
.block,
.row,
.column,
.interface-section,
.centered-container,
#header_column,
#query_input,
#synthesized_summary_output,
#fusion_output,
#user_token_input,
#tier_status_display,
#memory_log,
#memory_display_panel,
#pro_features_row {
    width: 100% !important;
    max-width: 100% !important;
    box-sizing: border-box !important;
    overflow-wrap: break-word !important;
}

textarea, input, button {
    max-width: 100% !important;
}

/* ✅🧠 CORRECTED: Force logo image to grayscale. Emojis (text) are unaffected. */
img {
    max-width: 100%;
    height: auto;
    display: block;
    filter: grayscale(100%);
}

/* ✅ FIX TEXTBOX TRUNCATION */
textarea {
    resize: vertical !important;
}

/* ✅ UNIFY FONT SIZE & TIGHTEN UI ON SMALL DEVICES */
@media screen and (max-width: 480px) {
    html, body {
        font-size: 15px !important;
    }
    
    /* 🔧 TIGHTEN GRADIO UI SCALE ON MOBILE */
    .gradio-container,
    .block,
    .row,
    .column,
    textarea,
    input,
    button {
        transform: scale(0.95);
        transform-origin: top center;
    }

    h1, h2, h3 {
        font-size: 1.2rem !important;
    }

    #header_logo img {
        max-height: 80px !important;
        padding: 8px !important;
    }

    #footer_attribution {
        font-size: 0.65rem !important;
    }
}

/* Loading Indicator Styles */
.cosmic-loading { display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 20px; text-align: center; }
.orbital-spinner { width: 40px; height: 40px; border: 4px solid var(--text-color-medium); border-top: 4px solid var(--text-color-light); border-radius: 50%; animation: spin 1s linear infinite; margin-bottom: 15px; }
@keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
.thinking-text { font-family: var(--font-body); color: var(--text-color-light); font-size: 0.9rem; }

/* Authentication Row and Tier Status Display */
#auth_row { margin-bottom: 1rem; align-items: center; }
#user_token_input textarea {
    background-color: var(--input-bg) !important; color: var(--text-color-light) !important;
    border: 1px solid var(--input-border) !important; border-radius: 8px !important;
    padding: 0.75em !important;
}
#tier_status_display {
    padding: 0.60rem 0.5rem !important; text-align: left; font-size: 0.9rem;
    color: var(--text-color-medium); display: flex; align-items: flex-start;
    min-height: 40px; line-height: 1.4;
}
#tier_status_display p { margin: 0 !important; }
#tier_status_display small { font-size: 0.9em; opacity: 0.9; }

/* Footer */
#footer_attribution {
    font-size: 0.75rem; color: var(--text-color-medium); text-align: center;
    margin-top: 2.5rem; padding-bottom: 1rem; opacity: 0.8; line-height: 1.5;
}
#footer_attribution strong { font-weight: 600; color: var(--text-color-light); }

/* Toast / Notification Styling */
.gradio-toast {
    background-color: #1e1e1e !important;
    color: var(--text-color-light) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 6px !important;
    box-shadow: 0 4px 12px rgba(0,0,0,0.3) !important;
}
.gradio-toast .toast-body {
    color: var(--text-color-light) !important;
}
.gradio-toast .toast-body svg { /* Style the icon if present */
    fill: var(--text-color-light) !important;
}
.gradio-toast.toast-info, .gradio-toast.toast-error, .gradio-toast.toast-warning, .gradio-toast.toast-success {
    /* General styles already applied, these are for any minor specific tweaks if needed */
}

/* Scrollable Output Boxes CSS */
#fusion_output, #synthesized_summary_output {
    max-height: 70vh;
    overflow-y: auto !important;
    padding: 1em;
    scroll-behavior: smooth;
    border: 1px solid var(--border-color) !important;
    border-radius: 10px !important;
    background-color: var(--input-bg) !important;
    word-wrap: break-word;
    white-space: pre-wrap;
}

/* Memory Display Panel Scrollable CSS */
#memory_display_panel {
    font-size: 0.85rem;
    color: var(--text-color-medium);
    background-color: #0A0A0A; padding: 0.75em;
    border-radius: 6px; border: 1px solid var(--border-color); max-height: 400px; overflow-y: auto;
}
#memory_display_panel .memory-entry { margin-bottom: 1rem; padding: 0.75rem; border: 1px solid #333; border-radius: 6px; background-color: #101010; }
#memory_display_panel .memory-entry strong { color: var(--text-color-light); font-weight: bold; }
#memory_display_panel .memory-entry em { color: var(--text-color-medium); font-style: normal; }
#memory_display_panel .memory-entry .query-text,
#memory_display_panel .memory-entry .summary-text { display: block; white-space: pre-wrap; word-break: break-word; margin-top: 0.25em; background-color: #181818; padding: 0.3em 0.5em; border-radius: 4px; }

/* A general scrollable container class, if needed elsewhere */
.scrollable-container { max-height: 75vh; overflow-y: auto; }
"""

# --- render_memory_entries_as_html ---
def render_memory_entries_as_html(memory_entries: List[dict]) -> str:
    if not memory_entries: return "<div class='memory-entry'>No stored memory entries or memory disabled.</div>"
    html_blocks = [];
    for entry in reversed(memory_entries[-5:]):
        query = html.escape(entry.get("query", "N/A")); ts_iso = entry.get("metadata", {}).get("timestamp_iso", "Unknown");
        try: dt_obj = time.strptime(ts_iso, '%Y-%m-%dT%H:%M:%SZ'); formatted_ts = time.strftime('%Y-%m-%d %H:%M UTC', dt_obj)
        except: formatted_ts = html.escape(ts_iso)
        models_list = entry.get("metadata", {}).get("active_models_queried", []); models_str = html.escape(", ".join(models_list)) if models_list else "N/A"
        summary_raw = entry.get("metadata", {}).get("synthesized_summary_text", entry.get("response", "No summary.")[:500]); summary = html.escape(summary_raw[:300]) + ("..." if len(summary_raw) > 300 else "")
        html_blocks.append(f"<div class='memory-entry'><p><strong>🕒 TS:</strong> {formatted_ts}</p><p><strong>🧠 Models:</strong> {models_str}</p><div><strong>❓ Q:</strong><div class='query-text'>{query}</div></div><div><strong>📌 Sum:</strong><div class='summary-text'>{summary}</div></div></div>")
    return "\n".join(html_blocks) if html_blocks else "<div class='memory-entry'>No processable entries.</div>"

# --- Format Tier Status for UI ---
def format_tier_status_for_ui(tier_name: str) -> str:
    features = TIER_FEATURES.get(tier_name, TIER_FEATURES["error (auth module failed)"])
    status_md = f"**{features['display_name']}**\n"
    status_md += f"<small>Memory: {features['emoji_map']['memory']} | "
    status_md += f"Export: {features['emoji_map']['export']} | "
    model_limit_from_backend = "N/A"
    if ai_system and hasattr(ai_system, 'config') and isinstance(ai_system.config, dict) and \
       'TIER_CONFIG' in ai_system.config and isinstance(ai_system.config['TIER_CONFIG'], dict) and \
       tier_name in ai_system.config['TIER_CONFIG'] and isinstance(ai_system.config['TIER_CONFIG'][tier_name], dict):
        backend_limit = ai_system.config['TIER_CONFIG'][tier_name].get('model_limit', 'N/A')
        if isinstance(backend_limit, int) and backend_limit >= 999 :
            model_limit_from_backend = "All"
        else: model_limit_from_backend = str(backend_limit)
    status_md += f"Models: Up to {model_limit_from_backend}"
    status_md += "</small>"
    return status_md

# --- Update Tier Display and Features UI ---
def update_tier_display_and_features_ui(user_token_str: Optional[str]) -> Tuple:
    tier_name = get_user_tier(user_token_str if user_token_str else "")
    tier_features = TIER_FEATURES.get(tier_name, TIER_FEATURES["free"])
    logger.info(f"UI: Token changed. Tier: '{tier_name}'. Features: {tier_features}")
    formatted_status = format_tier_status_for_ui(tier_name)
    export_button_interactive = tier_features["export_enabled"]
    export_button_classes = "interactive_button_enabled" if export_button_interactive else ""
    return (gr.update(value=formatted_status), gr.update(interactive=export_button_interactive, elem_classes=export_button_classes))

# --- Clear All Fields Function ---
def clear_all_fields():
    logger.info("Clearing all input fields and output areas.")
    default_tier_name = get_user_tier("")
    default_tier_features = TIER_FEATURES.get(default_tier_name, TIER_FEATURES["free"])
    return (
        "", # query_input_component
        "", # user_token_input
        gr.update(value=format_tier_status_for_ui(default_tier_name)), # tier_status_display
        "Detailed perspectives will appear here...", # fused_response_output_component
        "Synthesized insight will appear here...", # synthesized_summary_component
        gr.update(value="**Status:** Awaiting Input"), # active_models_display_component
        gr.update(value="<div class='memory-entry'>No memory entries loaded yet.</div>"), # memory_display_panel_html
        gr.update(interactive=default_tier_features["export_enabled"],
                  elem_classes="interactive_button_enabled" if default_tier_features["export_enabled"] else ""),
        None
    )

# --- Function to prepare memory data for client-side download ---
async def export_user_memory_data_for_download(user_token_str: Optional[str]) -> Optional[str]:
    try:
        tier_name = get_user_tier(user_token_str if user_token_str else "")
        tier_f = TIER_FEATURES.get(tier_name, TIER_FEATURES["free"])

        if not tier_f["export_enabled"]:
            gr.Warning("🚫 Export Memory is a Pro Tier feature. Please upgrade your plan.")
            return None

        if not ai_system or not hasattr(ai_system, 'memory_bank'):
            gr.Error("⚠️ Memory system is not available in the backend.")
            return None

        memory_data = None
        if hasattr(ai_system.memory_bank, 'get_all_memories_for_export_async'):
            logger.info("Using asynchronous get_all_memories_for_export_async.")
            memory_data = await ai_system.memory_bank.get_all_memories_for_export_async()
        elif hasattr(ai_system.memory_bank, 'get_all_memories_for_export'):
            logger.info("Using synchronous get_all_memories_for_export.")
            memory_data = ai_system.memory_bank.get_all_memories_for_export()
        else:
            gr.Error("⚠️ Memory export data retrieval method not available in backend MemoryBank.")
            return None

        if not memory_data:
            gr.Info("ℹ️ No memory entries to export.")
            return None

        json_data = json.dumps(memory_data, indent=2)

        with tempfile.NamedTemporaryFile(delete=False, mode="w", suffix=".json", encoding="utf-8", prefix="zotheos_export_") as tmp_file:
            tmp_file.write(json_data)
            tmp_file_path = tmp_file.name

        logger.info(f"✅ Memory data prepared for client download: {tmp_file_path}")
        gr.Info(f"✅ Memory export ready. Download: {os.path.basename(tmp_file_path)}")
        return tmp_file_path
    except Exception as e:
        logger.error(f"❌ Error preparing memory export for client: {e}", exc_info=True)
        gr.Error(f"⚠️ Error preparing export: {str(e)}")
        return None

# --- Build Gradio Interface ---
def build_interface(logo_path_param: Optional[str], favicon_path_param: Optional[str]) -> gr.Blocks:
    # ✅🧠 DEFINITIVE FIX: Use a true neutral (grayscale) theme as the foundation.
    theme = gr.themes.Base(
        primary_hue=gr.themes.colors.neutral, # Changed from "slate"
        secondary_hue=gr.themes.colors.neutral, # Changed from "gray"
        neutral_hue=gr.themes.colors.neutral, # Changed from "gray"
        font=[gr.themes.GoogleFont("Lato"),"ui-sans-serif"],
        font_mono=[gr.themes.GoogleFont("Inconsolata"),"ui-monospace"]
    ).set(
        body_background_fill="var(--background-main)",
    )

    initial_tier_name = get_user_tier("")
    initial_tier_features = TIER_FEATURES.get(initial_tier_name, TIER_FEATURES["free"])

    with gr.Blocks(theme=theme, css=zotheos_base_css, title=APP_TITLE) as demo:
        with gr.Column(elem_classes="centered-container"):
            with gr.Column(elem_id="header_column"):
                if logo_path_param:
                    gr.Image(value=logo_path_param, elem_id="header_logo", show_label=False, container=False, interactive=False)
                gr.Markdown("Ethical Fusion AI for Synthesized Intelligence", elem_id="header_subtitle")
            gr.Markdown("### 💡 Fusing perspectives for deeper truth.", elem_id="welcome_message")

            with gr.Row(elem_id="auth_row", elem_classes="interface-section", equal_height=False):
                user_token_input = gr.Textbox(label="🔑 Access Token", placeholder="Enter token for tiered features", type="password", elem_id="user_token_input", scale=3, container=False)
                tier_status_display = gr.Markdown(value=format_tier_status_for_ui(initial_tier_name), elem_id="tier_status_display")

            with gr.Group(elem_classes="interface-section"):
                query_input_component = gr.Textbox(label="Your Inquiry:", placeholder="e.g., Analyze the ethical implications of AI in art...", lines=6, elem_id="query_input")
                status_indicator_component = gr.HTML(elem_id="status_indicator", visible=False)
                with gr.Row(elem_classes="button-row"):
                    submit_button_component = gr.Button("Process Inquiry", elem_id="submit_button", variant="primary", scale=2)
                    clear_button_component = gr.Button("Clear All", elem_id="clear_button", variant="secondary", scale=1)

            with gr.Group(elem_classes="interface-section"):
                synthesized_summary_component = gr.Markdown(elem_id="synthesized_summary_output", value="Synthesized insight will appear here...", visible=True)
                fused_response_output_component = gr.Markdown(elem_id="fusion_output", value="Detailed perspectives will appear here...", visible=True)

            with gr.Row(elem_id="pro_features_row", elem_classes="interface-section", visible=True):
                export_memory_button = gr.Button(
                    "⬇️ Export Memory Log",
                    interactive=initial_tier_features["export_enabled"],
                    elem_classes="interactive_button_enabled"
                )

            with gr.Accordion("💡 System Status & Active Models", open=False, elem_id="status_accordion"):
                 active_models_display_component = gr.Markdown("**Status:** Initializing... ", elem_id="active_models_info")
            with gr.Accordion("🧠 Recent Interaction Chronicle (Last 5)", open=False, elem_id="memory_viewer_section"):
                memory_display_panel_html = gr.HTML(elem_id="memory_display_panel", value="<div class='memory-entry'>No memory entries yet.</div>")

            gr.Markdown(
                """---
                ZOTHEOS © 2025 ZOTHEOS LLC. System Architect: David A. Garcia.
                Built with open-source models: **Qwen** (Alibaba/Tongyi Qianwen License), **Mistral** (Mistral AI/Apache 2.0), **Gemma** (Google/Gemma T.O.U).
                Powered by **llama-cpp-python** (MIT) & **Gradio** (Apache 2.0). Please see `LICENSE.txt` for full details.
                Ethical Fusion AI for Humanity.
                """, elem_id="footer_attribution"
            )

        async def process_query_wrapper_internal(query_text_internal: str, user_token_from_ui: Optional[str]) -> Tuple:
            current_tier_name = get_user_tier(user_token_from_ui if user_token_from_ui else "")
            tier_display_name = TIER_FEATURES.get(current_tier_name, TIER_FEATURES["free"])["display_name"]
            loading_html = f"<div class='cosmic-loading'><div class='orbital-spinner'></div><div class='thinking-text'>ZOTHEOS is synthesizing (Tier: {tier_display_name})...</div></div>"
            yield (
                gr.update(value=loading_html, visible=True),
                gr.update(value="Processing...", interactive=False),
                gr.update(interactive=False),
                gr.update(),
                gr.update(value=""),
                gr.update(value=""),
                gr.update(),
                gr.update(value="<div class='memory-entry'>Loading...</div>")
            )

            if not ai_system or initialization_error:
                error_msg_html = f"<div style='color: var(--text-color-light); font-weight: bold; padding: 10px; border: 1px solid var(--border-color); background-color: #1a1a1a; border-radius: 6px;'>🚫 Core System Unresponsive. Details: {html.escape(str(initialization_error or 'Unknown Reason'))}. Please check logs or restart.</div>"
                yield (
                    gr.update(value=error_msg_html,visible=True),
                    gr.update(value="Offline",interactive=False),
                    gr.update(interactive=True),
                    gr.update(),
                    gr.update(value="System error occurred."),
                    gr.update(value="Insight unavailable due to system error."),
                    gr.update(value="**Status:** `Offline - Critical Error`"),
                    gr.update(value="<div class='memory-entry'>Memory unavailable due to system error.</div>")
                ); return
            query = query_text_internal.strip();
            if not query:
                yield (
                    gr.update(value="",visible=False),
                    gr.update(value="🔍 Process",interactive=True),
                    gr.update(interactive=True),
                    gr.update(),
                    gr.update(value="Enter query."),
                    gr.update(value="Insight awaits..."),
                    gr.update(value="**Status:** Awaiting Input"),
                    gr.update(value="<div class='memory-entry'>No query entered.</div>")
                ); return

            fused_perspectives_content = ""; synthesized_summary_content = ""; rendered_memory_html = "<div class='memory-entry'>Memory processing...</div>"
            try:
                logger.info(f"UI: Query: '{query[:60]}...' Token: '{user_token_from_ui[:3] if user_token_from_ui else 'N/A'}...'")
                current_mode_val = DEFAULT_INFERENCE_PRESET_INTERFACE
                full_response_content = await ai_system.process_query_with_fusion(query, user_token=user_token_from_ui, fusion_mode_override=current_mode_val)

                overall_header_marker = "## 🧠 ZOTHEOS Fused Perspectives 🧠"
                final_insight_marker = "## ✨ ZOTHEOS Final Synthesized Insight ✨"
                detailed_perspectives_marker = "### 💬 Detailed Individual Perspectives"

                overall_header = ""
                content_after_overall_header = full_response_content

                if full_response_content.startswith(overall_header_marker):
                    first_double_newline = full_response_content.find("\n\n")
                    if first_double_newline != -1:
                        overall_header = full_response_content[:first_double_newline + 2]
                        content_after_overall_header = full_response_content[first_double_newline + 2:]
                    else:
                        overall_header = full_response_content
                        content_after_overall_header = ""

                idx_final_insight = content_after_overall_header.find(final_insight_marker)

                if idx_final_insight != -1 :
                    synthesized_summary_content = overall_header + content_after_overall_header[idx_final_insight:]
                    temp_content_before_insight = content_after_overall_header[:idx_final_insight]
                    idx_detailed_perspectives_in_prefix = temp_content_before_insight.rfind(detailed_perspectives_marker)

                    if idx_detailed_perspectives_in_prefix != -1:
                         fused_perspectives_content = overall_header + temp_content_before_insight[idx_detailed_perspectives_in_prefix:]
                    else:
                         fused_perspectives_content = overall_header + "Detailed perspectives are integrated within the main insight or not separately provided."
                elif full_response_content:
                    synthesized_summary_content = full_response_content
                    fused_perspectives_content = "See main insight above for details."
                else:
                    synthesized_summary_content = overall_header + "No insight generated."
                    fused_perspectives_content = overall_header + "No perspectives generated."

                tier_features_for_query = TIER_FEATURES.get(current_tier_name, TIER_FEATURES["free"])
                if tier_features_for_query["memory_enabled"]:
                    if ai_system and hasattr(ai_system, 'memory_bank') and ai_system.memory_bank:
                        memory_data_list = []
                        if hasattr(ai_system.memory_bank, 'retrieve_recent_memories_async'):
                            memory_data_list = await ai_system.memory_bank.retrieve_recent_memories_async(limit=10)
                        elif hasattr(ai_system.memory_bank, 'retrieve_recent_memories'):
                            possible_coro = ai_system.memory_bank.retrieve_recent_memories(limit=10)
                            if asyncio.iscoroutine(possible_coro): memory_data_list = await possible_coro
                            else: memory_data_list = possible_coro # type: ignore
                        rendered_memory_html = render_memory_entries_as_html(memory_data_list)
                    else: rendered_memory_html = "<div class='memory-entry'>Memory system error.</div>"
                else: rendered_memory_html = "<div class='memory-entry'>Memory logging disabled for tier.</div>"
            except Exception as e:
                synthesized_summary_content = f"⚠️ Anomaly during processing: {html.escape(type(e).__name__)}. Please check logs."; fused_perspectives_content=""
                rendered_memory_html = f"<div class='memory-entry'>Error during processing: {html.escape(str(e))}</div>"
                logger.error(f"❌ Anomaly during query processing: {e}", exc_info=True)

            active_models_md = "**Status:** `Ready`";
            if ai_system and hasattr(ai_system, 'get_status_report') and callable(ai_system.get_status_report):
                try:
                    status_report = await ai_system.get_status_report()
                    models_queried_for_this_request = status_report.get("models_last_queried_for_perspectives", [])
                    active_cores_str = ", ".join(models_queried_for_this_request) if models_queried_for_this_request else "N/A"
                    actual_models_used_count_display = len(models_queried_for_this_request)
                    active_models_md = f"**Active Cores (Used for Query):** `{active_cores_str}` ({actual_models_used_count_display} models) | **Status:** `Online`"
                except Exception as e_status:
                    logger.warning(f"Status report error: {e_status}"); active_models_md = "**Status:** `Report Unavailable`"

            yield (
                gr.update(value="", visible=False),
                gr.update(value="Process Inquiry", interactive=True),
                gr.update(interactive=True),
                gr.update(),
                gr.update(value=fused_perspectives_content, visible=True),
                gr.update(value=synthesized_summary_content, visible=True),
                gr.update(value=active_models_md),
                gr.update(value=rendered_memory_html)
            )

        clear_fn_outputs_ordered = [query_input_component, user_token_input, tier_status_display, fused_response_output_component, synthesized_summary_component, active_models_display_component, memory_display_panel_html, export_memory_button]
        submit_fn_outputs_ordered = [status_indicator_component, submit_button_component, clear_button_component, query_input_component, fused_response_output_component, synthesized_summary_component, active_models_display_component, memory_display_panel_html]

        submit_button_component.click(fn=process_query_wrapper_internal, inputs=[query_input_component, user_token_input], outputs=submit_fn_outputs_ordered, show_progress="hidden")
        clear_button_component.click(fn=clear_all_fields, inputs=[], outputs=clear_fn_outputs_ordered, show_progress="hidden")

        token_change_outputs = [tier_status_display, export_memory_button]
        user_token_input.change(fn=update_tier_display_and_features_ui, inputs=[user_token_input], outputs=token_change_outputs, show_progress="hidden")

        export_memory_button.click(fn=export_user_memory_data_for_download, inputs=[user_token_input], outputs=[])

    return demo

# --- Main Execution Block (for direct run of this file) ---
if __name__ == "__main__":
    logger.info("--- Initializing ZOTHEOS Gradio Interface (V11 - Direct Run) ---")

    logger.info("Building ZOTHEOS Gradio UI...")
    zotheos_interface = build_interface(logo_path_verified, favicon_path_verified)
    logger.info("✅ ZOTHEOS Gradio UI built.")

    logger.info("🚀 Launching ZOTHEOS Gradio application...")
    print("\n" + "="*60 + f"\n ZOTHEOS Interface ({APP_TITLE} - V11)\n" + "="*60)
    if initialization_error:
        print(f"‼️ WARNING: ZOTHEOS AI System failed to initialize fully: {type(initialization_error).__name__} - {initialization_error}")
        if 'get_user_tier' not in globals() or get_user_tier == get_user_tier_fallback:
             print("   Additionally, user authentication module might have failed to load. Using fallback tier logic.")
    elif not ai_system:
        print(f"‼️ WARNING: ZOTHEOS AI System object (ai_system) is None. Backend will not function.")
    else:
        print("✅ ZOTHEOS AI System appears initialized.")

    try:
        import llama_cpp # type: ignore
        print("✅ llama-cpp-python library available.")
    except ImportError:
        print("\n‼️ CRITICAL WARNING: 'llama-cpp-python' not found! ZOTHEOS backend will fail.\n   Install via: pip install llama-cpp-python --force-reinstall --upgrade --no-cache-dir --verbose")

    print("\nZOTHEOS Interface starting. Access via URL below.")
    print("="*60 + "\n")

    launch_kwargs = {
        "server_name": "0.0.0.0",
        "server_port": int(os.getenv("PORT", 7860)),
        "share": os.getenv("GRADIO_SHARE", "False").lower() == "true",
        "inbrowser": True,
        "show_api": False,
        "favicon_path": favicon_path_verified
    }

    zotheos_interface.queue().launch(**launch_kwargs)
    logger.info("--- ZOTHEOS Gradio Interface Shutdown ---")