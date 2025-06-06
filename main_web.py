# main_web.py — ZOTHEOS Hugging Face Entry Point

import logging
from zotheos_interface_public import (
    build_interface,
    logo_path_verified,
    favicon_path_verified,
    APP_TITLE
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("ZOTHEOS_WebApp_HF")

# --- CRUCIAL FINAL CHECK ---
# Ensure your model loading code (likely in another file like 'modules/main_fusion_public.py')
# has been updated to use hf_hub_download.
# The server will download the models, so your code must not look for a local 'models/' folder.
logger.info("Verifying model loading strategy for web deployment...")
# (This is just a log message, the actual code change is in your backend file)

# --- Build and Launch the App ---
logger.info(f"Building Gradio UI for '{APP_TITLE}'...")
# Build the interface by calling the function from your other script
zotheos_app = build_interface(logo_path_verified, favicon_path_verified)

logger.info("UI built. Preparing to launch on Hugging Face Spaces...")
# The .queue() is important for handling multiple users.
# The .launch() command without arguments is what Hugging Face expects.
# It will handle the server and networking for you.
zotheos_app.queue().launch()

logger.info("ZOTHEOS app has been launched by the Hugging Face environment.")