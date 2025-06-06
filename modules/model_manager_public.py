# FILE: modules/model_manager_public.py

import asyncio
import logging
import os
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Dict, Any, Optional, List, Callable, Coroutine

# --- Llama.cpp Python Backend Import & Debug ---
logger_import_debug = logging.getLogger("ModelManager_ImportDebug")
if not logger_import_debug.handlers: # Minimal setup for this specific logger
    _h = logging.StreamHandler(sys.stdout)
    _f = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    _h.setFormatter(_f)
    logger_import_debug.addHandler(_h)
    logger_import_debug.setLevel(logging.INFO)

# --- Start of FIX: Define LlamaMock unconditionally and manage Llama variable ---
class LlamaMock:
    def __init__(self, model_path: str = "mock_model_path", *args, **kwargs):
        # Use logger_import_debug as it's defined at this point
        logger_import_debug.error(f"LlamaMock initialized for model_path='{model_path}' with args: {args}, kwargs: {kwargs}. llama_cpp is not installed or importable.")
        self.model_path = model_path
        self.n_ctx_train = kwargs.get('n_ctx', 0) # Mock common attributes
        self.metadata = {}

    def create_chat_completion(self, messages: List[Dict[str,str]], *args, **kwargs) -> Dict[str, Any]:
        logger_import_debug.error(f"LlamaMock: create_chat_completion called for '{self.model_path}', but llama_cpp is unavailable.")
        return {
            "id": "chatcmpl-mock", "object": "chat.completion", "created": int(time.time()),
            "model": self.model_path,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": f"[Error: Llama.cpp backend not available for {self.model_path}]"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        }

    def __call__(self, prompt: str, *args, **kwargs) -> Dict[str, Any]: # For older direct call style
        logger_import_debug.error(f"LlamaMock: __call__ used for '{self.model_path}', but llama_cpp is unavailable.")
        return {
            "id": "cmpl-mock", "object": "text_completion", "created": int(time.time()),
            "model": self.model_path,
            "choices": [{"text": f"[Error: Llama.cpp backend not available for {self.model_path}]", "index": 0, "logprobs": None, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        }
    
    def get_metadata(self) -> Dict[str, Any]:
        return self.metadata

    def tokenizer(self) -> Any:
        class MockTokenizer:
            def encode(self, text: str) -> List[int]: return [0] * len(text)
            def decode(self, tokens: List[int]) -> str: return "".join([chr(t) for t in tokens if t < 256]) # Simplistic
        return MockTokenizer()

# Initialize variables that will hold the class to be used
Llama_imported_class: Any = LlamaMock  # Default to mock
LlamaCppError_imported_class: Any = Exception  # Default to base Exception
LLAMA_CPP_AVAILABLE: bool = False

try:
    from llama_cpp import Llama as RealLlama, LlamaCppError as RealLlamaCppSpecificError
    Llama_imported_class = RealLlama
    LlamaCppError_imported_class = RealLlamaCppSpecificError
    LLAMA_CPP_AVAILABLE = True
    logger_import_debug.info("🎉 Final llama-cpp-python status: AVAILABLE. Using REAL Llama class. Error catching for Llama.cpp specifics will use 'LlamaCppSpecificError'.")
except ImportError:
    logger_import_debug.warning("Could not import 'Llama' and 'LlamaCppSpecificError' from 'llama_cpp' directly. Trying simpler import...")
    try:
        from llama_cpp import Llama as RealLlamaOnly # Try again without specific error import (older versions)
        Llama_imported_class = RealLlamaOnly
        # LlamaCppError_imported_class remains Exception
        LLAMA_CPP_AVAILABLE = True
        logger_import_debug.warning("Note: 'LlamaCppError' not found at llama_cpp top-level. Using base 'Exception' for LlamaCpp errors.")
        logger_import_debug.info("🎉 Final llama-cpp-python status: AVAILABLE (LlamaCppError not found). Using REAL Llama class.")
    except ImportError:
        # Llama_imported_class remains LlamaMock, LLAMA_CPP_AVAILABLE remains False
        logger_import_debug.critical("CRITICAL FAILURE: Cannot import 'Llama' from 'llama_cpp': No module named 'llama_cpp'. Using LlamaMock. Model loading will fail for real models.")
except Exception as e_import_general:
    # Llama_imported_class remains LlamaMock, LLAMA_CPP_AVAILABLE remains False
    logger_import_debug.critical(f"CRITICAL FAILURE: Unexpected error during 'llama_cpp' import: {e_import_general}. Using LlamaMock.", exc_info=True)

# Assign to the names used throughout the rest of this module
Llama = Llama_imported_class
LlamaCppError = LlamaCppError_imported_class
# --- End of FIX ---
# --- End Llama.cpp Python Backend Import & Debug ---


try:
    # Assume config_settings_public.py is in the same 'modules' directory or project root is in sys.path
    from modules.config_settings_public import (
        MODEL_PATHS, MODEL_SPECIFIC_PARAMS, INFERENCE_PRESETS, N_CTX_FALLBACK, VERBOSE_LLAMA_CPP
    )
except ImportError as e_conf:
    logging.critical(f"CRITICAL IMPORT ERROR in ModelManager: Cannot import from config_settings_public: {e_conf}")
    # Provide minimal fallbacks if config cannot be loaded, though this is a critical failure
    MODEL_PATHS = {}
    MODEL_SPECIFIC_PARAMS = {"_default": {"n_ctx": 2048, "n_gpu_layers": -1, "verbose": False}}
    INFERENCE_PRESETS = {"default": {"temperature": 0.7}}
    N_CTX_FALLBACK = 2048
    VERBOSE_LLAMA_CPP = False


logger = logging.getLogger("ZOTHEOS_ModelManager")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s] - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# Timeout for model generation tasks when run in a thread
GENERATION_THREAD_TIMEOUT_SECONDS = 300 # 5 minutes

class ModelManager:
    def __init__(self, device_preference: Optional[str] = "cuda", max_model_count: int = 1, max_ram_models_gb: float = 8.0):
        self.device_preference = device_preference
        self.loaded_models: Dict[str, Any] = {} # Stores Llama instances
        self.model_load_status: Dict[str, str] = {} # "unloaded", "loading", "loaded", "error"
        self.model_ram_estimate_gb: Dict[str, float] = {}
        self.total_estimated_ram_gb: float = 0.0
        self.max_model_count = max_model_count
        self.max_ram_models_gb = max_ram_models_gb
        self.model_load_order: List[str] = [] # Tracks order for LRU
        self.model_locks: Dict[str, asyncio.Lock] = {} # Per-model lock for loading
        self.executor = ThreadPoolExecutor(max_workers=max_model_count + 1, thread_name_prefix="LLM_Gen") # +1 for summarizer

        dev_log = "CPU (default)"
        if self.device_preference == "cuda":
            # Check NVIDIA drivers and CUDA toolkit availability (conceptual)
            try:
                import torch
                if torch.cuda.is_available():
                    dev_log = f"CUDA (preference): {torch.cuda.get_device_name(0)}"
                else:
                    dev_log = "CUDA (preference, but torch.cuda.is_available()=False)"
            except ImportError:
                dev_log = "CUDA (preference, but PyTorch not found for detailed check)"
            except Exception as e_cuda_check:
                dev_log = f"CUDA (preference, error during torch check: {e_cuda_check})"
        # Actual CUDA usage is determined by n_gpu_layers > 0 during Llama init

        logger.info(f"🔥 MMgr init. Dev Preference: {dev_log}. MaxRAM Models: {self.max_ram_models_gb}GB. Max Count: {self.max_model_count}.")
        if not LLAMA_CPP_AVAILABLE:
            logger.critical("❌ Llama.cpp backend CRITICAL FAILURE: The 'Llama' class could not be imported from 'llama_cpp'. ModelManager cannot load GGUF models. Please check installation and logs from 'ModelManager_ImportDebug'.")
        logger.info(f"✅ MMGR Config: Max Models: {self.max_model_count}, Max RAM: {self.max_ram_models_gb} GB.")

    async def _ensure_model_loaded(self, model_name: str) -> bool:
        if model_name not in self.model_locks:
            self.model_locks[model_name] = asyncio.Lock()

        async with self.model_locks[model_name]:
            if self.model_load_status.get(model_name) == "loaded" and model_name in self.loaded_models:
                if model_name in self.model_load_order: self.model_load_order.remove(model_name)
                self.model_load_order.append(model_name) # Move to end of LRU
                return True

            if self.model_load_status.get(model_name) == "loading":
                logger.info(f"Model '{model_name}' is already being loaded by another task. Waiting...")
                while self.model_load_status.get(model_name) == "loading": await asyncio.sleep(0.5)
                return self.model_load_status.get(model_name) == "loaded"

            self.model_load_status[model_name] = "loading"
            logger.info(f"Attempting to load model '{model_name}'...")

            await self._evict_models_if_needed(new_model_name_to_load=model_name)

            model_instance, ram_gb = await asyncio.to_thread(self._load_model_from_disk, model_name)
            if model_instance:
                self.loaded_models[model_name] = model_instance
                self.model_ram_estimate_gb[model_name] = ram_gb
                self.total_estimated_ram_gb += ram_gb
                self.model_load_status[model_name] = "loaded"
                if model_name in self.model_load_order: self.model_load_order.remove(model_name)
                self.model_load_order.append(model_name)
                logger.info(f"✅ Model '{model_name}' loaded successfully. RAM used by this model: {ram_gb:.2f}GB. Total est. RAM: {self.total_estimated_ram_gb:.2f}GB.")
                return True
            else:
                self.model_load_status[model_name] = "error"
                logger.error(f"❌ Failed to load model '{model_name}'.")
                # Clean up RAM if an estimate was added prematurely or if partial load occurred
                if model_name in self.model_ram_estimate_gb:
                    self.total_estimated_ram_gb -= self.model_ram_estimate_gb.pop(model_name)
                return False

    def _load_model_from_disk(self, model_name: str) -> tuple[Optional[Any], float]:
        # The check Llama == LlamaMock now works because LlamaMock is always defined
        if not LLAMA_CPP_AVAILABLE or Llama == LlamaMock: # Llama is the variable pointing to the class
            logger.error(f"Cannot load model '{model_name}': Llama.cpp backend not available (LLAMA_CPP_AVAILABLE={LLAMA_CPP_AVAILABLE}, Llama is LlamaMock: {Llama == LlamaMock}).")
            return None, 0.0

        model_path = MODEL_PATHS.get(model_name)
        if not model_path or not os.path.exists(model_path):
            logger.error(f"Model path for '{model_name}' not found or invalid: {model_path}")
            return None, 0.0

        specific_params = MODEL_SPECIFIC_PARAMS.get(model_name, {})
        default_params = MODEL_SPECIFIC_PARAMS.get("_default", {})

        load_params = default_params.copy()
        load_params.update(specific_params)
        load_params['model_path'] = model_path
        load_params.setdefault('verbose', VERBOSE_LLAMA_CPP) # Use global verbose if not set per model

        if 'n_ctx' not in load_params or load_params['n_ctx'] == 0: # Ensure n_ctx is valid
            load_params['n_ctx'] = N_CTX_FALLBACK
            logger.warning(f"n_ctx for model {model_name} was invalid or not found, using fallback: {N_CTX_FALLBACK}")

        logger.info(f"🔄 Loading '{model_name}' from: {model_path}")
        logger.debug(f"--- FINAL LOAD PARAMS FOR {model_name.upper()}: {load_params} ---")

        start_time = time.perf_counter()
        try:
            model_instance = Llama(**load_params) # Llama here is the variable assigned above
            load_time = time.perf_counter() - start_time
            logger.info(f"✅ Model '{model_name}' (Llama instance) initialized by Llama(...). Load time: {load_time:.2f}s.")

            ram_gb = os.path.getsize(model_path) / (1024**3)
            ram_gb_with_buffer = ram_gb * 1.5 + 1 # Heuristic: file size + 50% + 1GB for KV cache, etc.
            logger.info(f"Model '{model_name}' path: '{model_path}'. Est. RAM for this model (with buffer): {ram_gb_with_buffer:.2f}GB.")
            return model_instance, ram_gb_with_buffer
        except LlamaCppError as e_llama: # Catch specific LlamaCpp errors
            logger.error(f"❌ LlamaCppError loading model '{model_name}': {e_llama}", exc_info=True)
        except Exception as e:
            logger.error(f"❌ Generic error loading model '{model_name}': {e}", exc_info=True)
        return None, 0.0

    async def _evict_models_if_needed(self, new_model_name_to_load: Optional[str] = None):
        new_model_ram_gb_estimate = 0.0
        if new_model_name_to_load:
             model_path = MODEL_PATHS.get(new_model_name_to_load)
             if model_path and os.path.exists(model_path):
                 new_model_ram_gb_estimate = (os.path.getsize(model_path) / (1024**3)) * 1.5 + 1 # Same heuristic

        # Check if eviction is needed:
        # 1. If we are at max model count AND the new model isn't already loaded.
        # 2. OR if adding the new model would exceed total RAM limit AND we have models loaded.
        while (len(self.loaded_models) >= self.max_model_count and (new_model_name_to_load not in self.loaded_models)) or \
              (self.total_estimated_ram_gb + new_model_ram_gb_estimate > self.max_ram_models_gb and self.model_load_order):
            if not self.model_load_order:
                logger.warning("Eviction needed but model_load_order is empty. This should not happen if loaded_models is populated.")
                break

            model_to_evict = self.model_load_order.pop(0) # Evict LRU
            if model_to_evict in self.loaded_models:
                logger.warning(f"⚠️ Evicting model '{model_to_evict}' due to resource limits (Count: {len(self.loaded_models)}/{self.max_model_count}, RAM: {self.total_estimated_ram_gb:.2f}/{self.max_ram_models_gb:.2f}GB).")
                del self.loaded_models[model_to_evict] # Release model instance
                evicted_ram = self.model_ram_estimate_gb.pop(model_to_evict, 0)
                self.total_estimated_ram_gb -= evicted_ram
                self.model_load_status[model_to_evict] = "unloaded"
                logger.info(f"Model '{model_to_evict}' unloaded. RAM reclaimed: {evicted_ram:.2f}GB. Total est. RAM: {self.total_estimated_ram_gb:.2f}GB.")
            else:
                logger.warning(f"Model '{model_to_evict}' was in load order but not in loaded_models dict. Inconsistency detected.")


    def _generation_task_sync(self, model_instance: Any, prompt_messages: List[Dict[str,str]], gen_params: Dict[str, Any]) -> str:
        model_log_name = model_instance.model_path.split(os.sep)[-1] if hasattr(model_instance, 'model_path') else 'UnknownModel'
        logger.debug(f"--- [{model_log_name}] ENTERING _generation_task_sync ---")
        output_text = "[Error: Generation failed in thread]"
        try:
            completion = model_instance.create_chat_completion(messages=prompt_messages, **gen_params)
            if completion and "choices" in completion and completion["choices"]:
                message_content = completion["choices"][0].get("message", {}).get("content")
                output_text = message_content.strip() if message_content else "[No content in choice message]"
            else:
                output_text = "[No choices in completion or completion is empty]"
                logger.warning(f"[{model_log_name}] Malformed completion object: {completion}")
            logger.debug(f"✅ [{model_log_name}] Sync generation successful. Preview: {output_text[:100].replace(chr(10),' ')}...")
        except Exception as e:
            logger.error(f"❌ [{model_log_name}] Exception during model generation in thread: {e}", exc_info=True)
            output_text = f"[Error: Exception during generation - {type(e).__name__}: {str(e)[:100]}]"
        logger.debug(f"--- [{model_log_name}] EXITING _generation_task_sync ---")
        return output_text

    async def async_generation_wrapper(self, model_instance: Any, prompt_messages: List[Dict[str,str]], gen_params: Dict[str, Any], model_name_for_log: str) -> str:
        logger.debug(f"--- [{model_name_for_log}] ENTERING async_generation_wrapper ---")
        output_str = f"[Error: Model '{model_name_for_log}' instance not available for generation]"
        if not model_instance:
            logger.error(f"[{model_name_for_log}] Model instance is None in async_generation_wrapper.")
            return output_str
        try:
            output_str = await asyncio.wait_for(
                asyncio.to_thread(self._generation_task_sync, model_instance, prompt_messages, gen_params),
                timeout=GENERATION_THREAD_TIMEOUT_SECONDS + 10
            )
        except asyncio.TimeoutError:
            logger.error(f"❌ [{model_name_for_log}] Generation task TIMED OUT in asyncio.wait_for (>{GENERATION_THREAD_TIMEOUT_SECONDS + 10}s).")
            output_str = f"[Error: Generation timed out for model {model_name_for_log}]"
        except Exception as e:
            logger.error(f"❌ [{model_name_for_log}] Exception in async_generation_wrapper: {e}", exc_info=True)
            output_str = f"[Error: Async wrapper exception for model {model_name_for_log} - {type(e).__name__}]"
        logger.debug(f"--- [{model_name_for_log}] EXITING async_generation_wrapper ---")
        return output_str


    async def generate_with_model(self, model_name: str, prompt: str, preset_name: str = "default", system_prompt: Optional[str] = None) -> str:
        logger.info(f"--- [{model_name}] Received generation request. System prompt: {'Yes' if system_prompt else 'No'}. Preset: {preset_name} ---")
        # Check if Llama class is LlamaMock (meaning llama.cpp is not available)
        if not LLAMA_CPP_AVAILABLE or Llama == LlamaMock:
            logger.error(f"Cannot generate with model '{model_name}': Llama.cpp backend not available (LLAMA_CPP_AVAILABLE={LLAMA_CPP_AVAILABLE}, Llama is LlamaMock: {Llama == LlamaMock}).")
            return "[Error: Llama.cpp backend core ('Llama' class) not available or is mocked]"

        if not await self._ensure_model_loaded(model_name):
            return f"[Error: Model '{model_name}' could not be loaded or is not available.]"

        model_instance = self.loaded_models.get(model_name)
        if not model_instance:
            logger.error(f"Model '{model_name}' instance not found in loaded_models after successful _ensure_model_loaded. This is unexpected.")
            return f"[Error: Model '{model_name}' instance unexpectedly not found after load attempt.]"

        gen_params = INFERENCE_PRESETS.get(preset_name, INFERENCE_PRESETS.get("balanced", {"temperature": 0.7})).copy() # Fallback to "balanced" then hardcoded default

        prompt_messages = []
        if system_prompt and system_prompt.strip():
            prompt_messages.append({"role": "system", "content": system_prompt})
        prompt_messages.append({"role": "user", "content": prompt})

        logger.debug(f"[{model_name}] Calling async_generation_wrapper with messages: {prompt_messages}, params: {gen_params}")
        response_text = await self.async_generation_wrapper(model_instance, prompt_messages, gen_params, model_name)
        logger.info(f"--- [{model_name}] Generation complete. Response length: {len(response_text)} ---")
        return response_text

    def get_loaded_model_stats(self) -> Dict[str, Any]:
        return {
            "loaded_model_count": len(self.loaded_models),
            "total_ram_estimate_gb": round(self.total_estimated_ram_gb, 2),
            "max_ram_models_gb": self.max_ram_models_gb,
            "max_model_count": self.max_model_count,
            "models_in_memory": list(self.loaded_models.keys()),
            "load_order_lru": self.model_load_order, # LRU (oldest) to MRU (newest)
            "model_ram_details_gb": {name: round(ram, 2) for name, ram in self.model_ram_estimate_gb.items()},
            "model_load_status": self.model_load_status,
            "llama_cpp_available": LLAMA_CPP_AVAILABLE,
            "llama_class_is_mock": Llama == LlamaMock
        }

    def shutdown(self):
        logger.info("🔌 Shutting down ModelManager...")
        self.executor.shutdown(wait=True, cancel_futures=True) # Cancel pending futures on shutdown
        self.loaded_models.clear()
        self.model_ram_estimate_gb.clear()
        self.model_load_order.clear()
        self.model_load_status.clear()
        self.total_estimated_ram_gb = 0
        logger.info("✅ ModelManager shutdown complete.")

# Example usage for direct testing
async def main_model_manager_test():
    # Configure logging for the test
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s] - %(message)s')
    # Set specific loggers to higher levels if too verbose
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    
    logger.info("--- ModelManager Test ---")

    if not MODEL_PATHS:
        logger.error("MODEL_PATHS is empty. Cannot run test. Check config_settings_public.py")
        return
    if Llama == LlamaMock: # Check if we are using the mock
        logger.warning("Llama.cpp is not available. Running test with LlamaMock. Responses will be mocked.")

    # Example: max 2 models, 10GB RAM limit
    mm = ModelManager(max_model_count=2, max_ram_models_gb=10.0)
    
    test_models_available = list(MODEL_PATHS.keys())
    if not test_models_available:
        logger.error("No models defined in MODEL_PATHS for testing.")
        mm.shutdown()
        return

    # Select up to 3 models for testing, or fewer if not available
    model1_name = test_models_available[0]
    model2_name = test_models_available[1] if len(test_models_available) > 1 else model1_name
    model3_name = test_models_available[2] if len(test_models_available) > 2 else model1_name
    
    test_queries = [
        (model1_name, "What is the capital of France?", "precise", "You are a helpful geography expert."),
        (model2_name, "Explain black holes simply.", "creative", None),
        (model1_name, "What is 1+1?", "precise", None), # Query model1 again
    ]
    if len(test_models_available) > 2 and model3_name not in [model1_name, model2_name]: # Only add model3 if it's distinct and available
        test_queries.append((model3_name, "Tell me a short joke.", "default", None))

    for i, (model_name, query, preset, sys_prompt) in enumerate(test_queries):
        logger.info(f"\n--- Test Query {i+1}: Model '{model_name}', Preset '{preset}' ---")
        response = await mm.generate_with_model(model_name, query, preset_name=preset, system_prompt=sys_prompt)
        print(f"Response from '{model_name}': {response[:150]}...") # Print more of the response
        print(f"Stats after '{model_name}': {mm.get_loaded_model_stats()}")
        await asyncio.sleep(1) # Small delay between queries

    mm.shutdown()
    logger.info("--- ModelManager Test Complete ---")

if __name__ == "__main__":
    # This allows testing ModelManager directly if needed
    # Ensure your MODEL_PATHS in config_settings_public.py point to valid GGUF files
    # To run: python -m modules.model_manager_public
    asyncio.run(main_model_manager_test())