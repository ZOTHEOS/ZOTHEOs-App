import logging
import asyncio
from typing import Dict, Any, Type

logger = logging.getLogger(__name__)

# ✅ Define Pre-Approved Plugins (Only Trusted Plugins for Public Use)
ALLOWED_PLUGINS = {
    "basic_math": "BasicMathPlugin",  # Example of a basic utility plugin
    "date_time": "DateTimePlugin"     # Example of a simple utility plugin
}

class PluginManager:
    """🚀 Public Plugin Manager (Pre-Approved Plugins Only)"""

    def __init__(self):
        self.plugins: Dict[str, Any] = {}

    async def load_plugin(self, plugin_name: str):
        """✅ Load Pre-Approved Plugin."""
        if plugin_name not in ALLOWED_PLUGINS:
            logger.warning(f"⚠️ Plugin '{plugin_name}' is not available in the public version.")
            return f"⚠️ Plugin '{plugin_name}' not allowed in public version."

        if plugin_name in self.plugins:
            logger.info(f"✅ Plugin '{plugin_name}' already loaded.")
            return f"✅ Plugin '{plugin_name}' already loaded."

        try:
            # ✅ Use pre-defined plugin mappings for security
            plugin_class = globals().get(ALLOWED_PLUGINS[plugin_name])
            if plugin_class:
                plugin_instance = plugin_class()
                self.plugins[plugin_name] = plugin_instance
                logger.info(f"✅ Plugin '{plugin_name}' loaded successfully.")
                return f"✅ Plugin '{plugin_name}' loaded successfully."
            else:
                logger.error(f"❌ Plugin class '{plugin_name}' not found.")
                return f"❌ Plugin class '{plugin_name}' not found."
        except Exception as e:
            logger.exception(f"❌ Error loading plugin '{plugin_name}': {e}")
            return f"❌ Error loading plugin '{plugin_name}'."

    async def execute_plugin_command(self, plugin_name: str, command: str, args: Dict[str, Any] = {}):
        """✅ Execute Plugin Command (Predefined Only)."""
        if plugin_name not in self.plugins:
            logger.warning(f"⚠️ Attempt to execute command from unloaded plugin '{plugin_name}'.")
            return f"⚠️ Plugin '{plugin_name}' not loaded."

        plugin = self.plugins[plugin_name]
        if hasattr(plugin, command):
            method = getattr(plugin, command)
            try:
                if asyncio.iscoroutinefunction(method):
                    return await method(**args)
                return method(**args)
            except Exception as e:
                logger.error(f"❌ Error executing command '{command}' in plugin '{plugin_name}': {e}")
                return f"❌ Error executing '{command}' in '{plugin_name}'."
        else:
            logger.warning(f"⚠️ Command '{command}' not found in plugin '{plugin_name}'.")
            return f"⚠️ Command '{command}' not found in '{plugin_name}'."

# ✅ Example Predefined Plugin Classes
class BasicMathPlugin:
    """Basic Math Plugin (Example)"""
    
    def add(self, x, y):
        return x + y
    
    def subtract(self, x, y):
        return x - y

class DateTimePlugin:
    """Date/Time Plugin (Example)"""
    
    async def current_time(self):
        from datetime import datetime
        return datetime.utcnow().isoformat()

