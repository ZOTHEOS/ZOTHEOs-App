# FILE: modules/memory_bank_public.py (Finalized for Beta Checklist & Async Correctness)

import os
import json
import time 
import asyncio
import logging
import traceback # For more detailed error logging if needed
from typing import List, Dict, Optional, Any
from json import JSONDecodeError
from datetime import datetime, timezone # For ISO timestamps
from pathlib import Path # For home directory in export
import sys # For logger setup if run standalone

logger = logging.getLogger("ZOTHEOS_MemoryBank")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s] - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# --- Define memory file paths and limits ---
try:
    # Path logic for when memory_bank_public.py is in 'modules' folder
    # and data directory is in project root ('../zotheos_public_data')
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    project_root_dir = os.path.dirname(current_file_dir) # This is ZOTHEOS_Release_Package
    
    # Check if running from PyInstaller bundle
    if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
        # If bundled, data directory is relative to sys._MEIPASS (the temp extraction folder)
        DATA_BASE_DIR = os.path.join(sys._MEIPASS, "zotheos_public_data")
    else:
        # If running as script, data directory is relative to project root
        DATA_BASE_DIR = os.path.join(project_root_dir, "zotheos_public_data")

    if not os.path.exists(DATA_BASE_DIR):
        os.makedirs(DATA_BASE_DIR, exist_ok=True)
        logger.info(f"Created data base directory: {DATA_BASE_DIR}")
        
    MEMORY_DIR = os.path.join(DATA_BASE_DIR, "zotheos_memory")
    if not os.path.exists(MEMORY_DIR):
        os.makedirs(MEMORY_DIR, exist_ok=True)
        logger.info(f"Created memory directory: {MEMORY_DIR}")
        
except Exception as e:
     logger.critical(f"❌ Failed to setup base/memory directory for MemoryBank: {e}. Using fallback in user home.")
     MEMORY_DIR = os.path.join(os.path.expanduser("~"), ".zotheos", "zotheos_memory")
     os.makedirs(MEMORY_DIR, exist_ok=True)


MEMORY_FILE_PATH = os.path.join(MEMORY_DIR, "zotheos_memory.json")
MEMORY_FILE_TMP = os.path.join(MEMORY_DIR, "zotheos_memory_tmp.json")
MEMORY_SIZE_LIMIT = 1000 
MEMORY_SCHEMA_VERSION = 1.2 # Bumped for metadata structure in entry

class MemoryBank:
    def __init__(self):
        self.memory_list: List[Dict[str, Any]] = []
        self.memory_dict: Dict[str, Dict[str, Any]] = {} # For ID-based lookups, ID should be string
        self.next_id = 0
        logger.info(f"🧠 Initializing Memory Bank. Memory file: {MEMORY_FILE_PATH}")
        self._load_memory() # Load initial state
        
        if self.memory_list:
            try:
                # Ensure IDs are treated as integers for max() if they are numeric strings
                numeric_ids = [int(m.get('id', -1)) for m in self.memory_list if str(m.get('id', '')).isdigit()]
                if numeric_ids:
                    max_id = max(numeric_ids)
                    self.next_id = max(max_id + 1, len(self.memory_list))
                else: # No numeric IDs found
                    self.next_id = len(self.memory_list)
            except ValueError: # Fallback if conversion to int fails for some reason
                self.next_id = len(self.memory_list) 
            logger.info(f"Loaded {len(self.memory_list)} memories. Next ID set to {self.next_id}.")
        else:
            logger.info("Initialized with empty memory.")

    def _reset_memory_state(self):
        self.memory_list = []
        self.memory_dict = {}
        self.next_id = 0
        logger.info("Memory state has been reset.")

    def _load_memory(self):
        logger.info(f"Attempting to load memory from {MEMORY_FILE_PATH}...")
        try:
            if os.path.exists(MEMORY_FILE_PATH):
                with open(MEMORY_FILE_PATH, "r", encoding="utf-8") as file:
                    data = json.load(file)
                if isinstance(data, dict) and "entries" in data and isinstance(data["entries"], list):
                    self.memory_list = data["entries"]
                    # Rebuild dictionary, ensuring IDs are strings for keys
                    self.memory_dict = {str(m['id']): m for m in self.memory_list if m.get('id') is not None}
                    logger.info(f"✅ Successfully loaded {len(self.memory_list)} memory entries (schema version: {data.get('schema_version', 'Unknown')}).")
                elif isinstance(data, list): 
                    logger.warning(f"Old memory format (list) detected. Converting and saving in new format.")
                    self.memory_list = data
                    self.memory_dict = {str(m.get('id','')): m for m in self.memory_list if m.get('id') is not None}
                    self._save_memory() # Save immediately in new format
                else:
                    logger.warning(f"⚠️ Memory file {MEMORY_FILE_PATH} has an unexpected main structure. Resetting memory.")
                    self._reset_memory_state()
            else:
                logger.info(f"✅ No existing memory file found at {MEMORY_FILE_PATH}. Starting fresh.")
                self._reset_memory_state()
        except JSONDecodeError as e:
            logger.error(f"❌ Error decoding JSON from memory file {MEMORY_FILE_PATH}: {e}. File might be corrupted. Resetting memory.", exc_info=False)
            self._handle_corrupted_memory_file()
        except FileNotFoundError: 
            logger.info(f"✅ Memory file {MEMORY_FILE_PATH} not found. Starting fresh (FileNotFound).")
            self._reset_memory_state()
        except Exception as e:
            logger.error(f"❌ Unexpected error loading memory: {e}. Resetting memory.", exc_info=True)
            self._reset_memory_state()

    def _handle_corrupted_memory_file(self):
        self._reset_memory_state()
        if os.path.exists(MEMORY_FILE_PATH):
            try:
                corrupt_backup_path = f"{MEMORY_FILE_PATH}.corrupt_{datetime.now().strftime('%Y%m%d%H%M%S')}"
                os.rename(MEMORY_FILE_PATH, corrupt_backup_path)
                logger.info(f"Backed up corrupted memory file to {corrupt_backup_path}")
            except OSError as backup_err:
                logger.error(f"Failed to backup corrupted memory file: {backup_err}")

    def _save_memory(self): # This is synchronous
        logger.debug(f"Attempting atomic save to {MEMORY_FILE_PATH}...")
        try:
            data_to_save = {"schema_version": MEMORY_SCHEMA_VERSION, "entries": self.memory_list}
            with open(MEMORY_FILE_TMP, "w", encoding="utf-8") as file:
                json.dump(data_to_save, file, indent=2) 
            os.replace(MEMORY_FILE_TMP, MEMORY_FILE_PATH)
            logger.info(f"✅ Memory saved successfully ({len(self.memory_list)} entries).")
        except Exception as e:
            logger.error(f"❌ Error saving memory: {e}", exc_info=True)
            if os.path.exists(MEMORY_FILE_TMP):
                try: os.remove(MEMORY_FILE_TMP)
                except OSError: pass
        finally:
            if os.path.exists(MEMORY_FILE_TMP):
                try: os.remove(MEMORY_FILE_TMP)
                except OSError as e_rem: logger.warning(f"Could not remove temp memory file {MEMORY_FILE_TMP}: {e_rem}")

    async def save_memory_async(self):
         logger.debug("Scheduling asynchronous memory save...")
         await asyncio.to_thread(self._save_memory)

    async def store_memory_async(self, query: str, response: str, metadata: Optional[Dict[str, Any]] = None):
        if not query or not response:
             logger.warning("⚠️ Attempted to store empty query or response. Skipping.")
             return
        try:
            current_id_num = self.next_id
            current_metadata = metadata.copy() if metadata is not None else {} # Work with a copy

            # Ensure a proper ISO-formatted timestamp is in metadata
            current_metadata['timestamp_iso'] = datetime.now(timezone.utc).isoformat()

            memory_entry = {
                'id': str(current_id_num), 
                'query': query,
                'response': response, # This is the full_fused_output from main_fusion
                'created_at_unix': time.time(), # Retain for fallback sorting
                'schema_version': MEMORY_SCHEMA_VERSION,
                'metadata': current_metadata # This now contains timestamp_iso and other details
            }
            self.next_id += 1
            self.memory_list.append(memory_entry)
            self.memory_dict[str(current_id_num)] = memory_entry
            logger.info(f"Stored memory entry ID {current_id_num}.")
            
            removed_count = 0
            while len(self.memory_list) > MEMORY_SIZE_LIMIT:
                oldest_memory = self.memory_list.pop(0)
                oldest_id = str(oldest_memory.get('id', ''))
                if oldest_id and oldest_id in self.memory_dict: del self.memory_dict[oldest_id]
                removed_count += 1
            if removed_count > 0: logger.info(f"Removed {removed_count} oldest entries for size limit.")
            await self.save_memory_async()
        except Exception as e: logger.error(f"❌ Error storing memory entry: {e}", exc_info=True)

    async def retrieve_recent_memories_async(self, limit: int = 5) -> List[Dict[str, Any]]:
        logger.debug(f"Retrieving up to {limit} recent memories, sorted.")
        if not self.memory_list: return []
        try:
            def get_sort_key(entry):
                ts_iso = entry.get('metadata', {}).get('timestamp_iso')
                # Fallback to created_at_unix if timestamp_iso is missing or unparsable
                if ts_iso:
                    try: return datetime.fromisoformat(ts_iso.replace('Z', '+00:00'))
                    except ValueError: 
                        logger.warning(f"Could not parse timestamp_iso '{ts_iso}' for entry ID {entry.get('id')}. Falling back to created_at_unix.")
                        pass # Fall through to use created_at_unix
                return datetime.fromtimestamp(entry.get('created_at_unix', 0), timezone.utc)

            # Make a copy for sorting to avoid modifying self.memory_list if other operations occur
            sorted_entries = sorted(list(self.memory_list), key=get_sort_key, reverse=True)
            
            actual_limit = max(0, min(limit, len(sorted_entries)))
            recent_sorted_memories = sorted_entries[:actual_limit]
            logger.info(f"Retrieved {len(recent_sorted_memories)} recent memories (sorted).")
            return recent_sorted_memories
        except Exception as e:
            logger.error(f"❌ Error retrieving and sorting recent memories: {e}", exc_info=True)
            # Fallback to unsorted last N if sorting fails
            return self.memory_list[-limit:][::-1] if limit > 0 and self.memory_list else []
            
    def load_all_memory_entries_structured(self) -> List[Dict[str, Any]]:
        logger.info(f"Loading all {len(self.memory_list)} memory entries (structured).")
        return list(self.memory_list)

    async def load_all_memory_entries_structured_async(self) -> List[Dict[str, Any]]:
        logger.info(f"Asynchronously loading all {len(self.memory_list)} memory entries (structured).")
        return list(self.memory_list) # For now, direct copy is fine as it's in-memory

    async def retrieve_memory_by_id(self, memory_id: str) -> Optional[Dict[str, Any]]:
        memory_id_str = str(memory_id)
        try:
            memory = self.memory_dict.get(memory_id_str)
            if memory: logger.info(f"Retrieved memory entry ID {memory_id_str}.")
            else: logger.warning(f"Memory ID {memory_id_str} not found in dictionary.")
            return memory 
        except Exception as e: logger.error(f"❌ Error retrieving memory ID {memory_id_str}: {e}", exc_info=True); return None

    async def retrieve_last_response(self) -> Optional[str]:
        if not self.memory_list: logger.info("Retrieve last response: Memory list is empty."); return None
        try:
            last_entry = self.memory_list[-1]; response = last_entry.get('response')
            if isinstance(response, str) and response.strip(): logger.info("Retrieve last response: Found valid response."); return response
            else: logger.warning(f"Retrieve last response: Last entry (ID {last_entry.get('id', 'N/A')}) has empty response."); return None
        except Exception as e: logger.error(f"❌ Error retrieving last response: {e}", exc_info=True); return None

    async def clear_all_memory(self):
        logger.warning("Initiating complete memory wipe...")
        try: self._reset_memory_state(); await self.save_memory_async(); logger.info("✅ All memory cleared successfully."); return True
        except Exception as e: logger.error(f"❌ Error clearing memory: {e}", exc_info=True); return False

    async def delete_memory_by_id(self, memory_id: str):
        logger.warning(f"Attempting to delete memory ID {memory_id}...")
        memory_id_str = str(memory_id)
        try:
            if memory_id_str in self.memory_dict:
                del self.memory_dict[memory_id_str]
                self.memory_list = [m for m in self.memory_list if str(m.get('id', '')) != memory_id_str]
                await self.save_memory_async()
                logger.info(f"✅ Memory with ID {memory_id_str} deleted successfully.")
                return True
            else:
                logger.warning(f"⚠️ Memory ID {memory_id_str} not found for deletion.")
                return False
        except Exception as e:
            logger.error(f"❌ Error deleting memory ID {memory_id_str}: {e}", exc_info=True)
            return False

    async def get_memory_stats(self) -> Dict[str, Any]: # This is now async
        logger.info("Calculating memory statistics...")
        stats: Dict[str, Any] = {'total_entries': len(self.memory_list), 'disk_usage_mb': 0.0, 'memory_limit': MEMORY_SIZE_LIMIT, 'next_id': self.next_id, 'schema_version': MEMORY_SCHEMA_VERSION }
        try:
            if os.path.exists(MEMORY_FILE_PATH):
                file_size_bytes = await asyncio.to_thread(os.path.getsize, MEMORY_FILE_PATH)
                stats['disk_usage_mb'] = round(file_size_bytes / (1024 * 1024), 3)
            logger.info(f"Memory Stats: {stats}")
        except Exception as e: logger.error(f"❌ Error calculating memory file size: {e}", exc_info=True)
        return stats

    def export_memory_to_file_sync(self) -> Optional[str]:
        """Synchronously exports memory file. Returns exported file path or None."""
        # NOTE (Future Personalization): Consider allowing user to choose export location via UI dialog.
        if not os.path.exists(MEMORY_FILE_PATH):
            logger.warning("No memory file to export because it doesn't exist.")
            return None
        if not self.memory_list: # Also check if there are any entries to export
            logger.warning("No memory entries to export, memory file might be empty or just schema.")
            # Decide if you want to export an empty "entries" file or return None
            # For now, let's allow exporting an empty structure.
            # return None 
            
        try:
            export_dir = Path.home() / "Desktop"
            if not (export_dir.exists() and export_dir.is_dir()):
                export_dir = Path.home() / "Downloads"
            if not (export_dir.exists() and export_dir.is_dir()):
                export_dir = Path(MEMORY_DIR) # Fallback
                logger.warning(f"Desktop/Downloads not found/accessible, exporting to memory directory: {export_dir}")
            os.makedirs(export_dir, exist_ok=True) # Ensure export_dir exists

            timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
            export_filename = f"zotheos_memory_export_{timestamp_str}.json"
            export_full_path = export_dir / export_filename

            import shutil 
            shutil.copy2(MEMORY_FILE_PATH, export_full_path)
            
            logger.info(f"✅ Memory successfully exported to: {export_full_path}")
            return str(export_full_path)
        except Exception as e:
            logger.error(f"❌ Failed to export memory: {e}", exc_info=True)
            return None

    async def export_memory_to_file_async(self) -> Optional[str]:
        """Asynchronously exports the memory file."""
        logger.info("Scheduling asynchronous memory export...")
        return await asyncio.to_thread(self.export_memory_to_file_sync)


async def main_test():
     logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s] - %(message)s')
     logger.info("--- MemoryBank Test ---")
     mb = MemoryBank()
     
     logger.info(f"Initial memory file path: {MEMORY_FILE_PATH}")
     if os.path.exists(MEMORY_FILE_PATH):
         logger.info("Memory file exists. Clearing for fresh test.")
         await mb.clear_all_memory()
     else:
         logger.info("No pre-existing memory file found for test.")

     stats1 = await mb.get_memory_stats()
     logger.info(f"Stats after potential clear: {stats1}")

     # Test Store
     await mb.store_memory_async("Query 1", "Response 1", metadata={"custom_field": "value1", "tier_at_interaction": "free"})
     await asyncio.sleep(0.01) 
     await mb.store_memory_async("Query 2", "Response 2", metadata={"user_token_used_prefix": "tes***", "synthesized_summary_text": "Summary for Q2"})
     await asyncio.sleep(0.01)
     await mb.store_memory_async("Query 3", "Response 3", metadata={"tier_at_interaction": "pro", "synthesized_summary_text": "Summary for Q3"})
     await asyncio.sleep(0.01)
     await mb.store_memory_async("Query 4", "Response 4", metadata={}) # No extra metadata
     await asyncio.sleep(0.01)
     await mb.store_memory_async("Query 5", "Response 5", metadata={"synthesized_summary_text": "This is summary 5."})
     await asyncio.sleep(0.01)
     await mb.store_memory_async("Query 6", "Response 6", metadata={"synthesized_summary_text": "This is summary 6, a bit longer than the preview."})

     recent_for_display = await mb.retrieve_recent_memories_async(limit=5)
     logger.info(f"Recent 5 (for display, should be newest first - Q6, Q5, Q4, Q3, Q2):")
     for i, item in enumerate(recent_for_display):
         ts = item.get('metadata',{}).get('timestamp_iso', item.get('created_at_unix'))
         logger.info(f"  {i+1}. ID: {item.get('id')}, Timestamp: {ts}, Query: {item.get('query')[:20]}..., Summary in meta: {'synthesized_summary_text' in item.get('metadata', {})}")

     stats2 = await mb.get_memory_stats()
     logger.info(f"Memory Stats after storing: {stats2}")

     exported_file = await mb.export_memory_to_file_async()
     if exported_file: logger.info(f"Test export successful: {exported_file}")
     else: logger.error("Test export failed.")
     
     logger.info("--- MemoryBank Test Complete ---")

if __name__ == "__main__":
     asyncio.run(main_test())