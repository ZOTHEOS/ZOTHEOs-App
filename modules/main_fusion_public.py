# FILE: modules/main_fusion_public.py (Finalized with Tier Logic, Async, Fusion Summary)

import asyncio
import logging
import os
import sys
import time
import json
from typing import Dict, Any, Optional, List, Union

try:
    # Correctly determine project_root assuming this file is in 'modules'
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
except Exception as e_path:
    # Basic logging if path setup fails, though critical
    logging.basicConfig(level=logging.ERROR)
    logging.error(f"Error setting up sys.path in main_fusion_public.py: {e_path}")

try:
    from modules.config_settings_public import (
        MODEL_PATHS, MAX_CONCURRENT_MODELS, MAX_RAM_MODELS_GB, # Used by ModelManager init
        DEFAULT_SYSTEM_PROMPT, SYSTEM_PERSONAS, INFERENCE_PRESETS, DEFAULT_INFERENCE_PRESET,
        MODEL_ROLES, MODEL_ROLE_SYSTEM_PROMPTS
    )
    from modules.model_manager_public import ModelManager
    from modules.memory_bank_public import MemoryBank
    from modules.user_auth import get_user_tier
except ImportError as e:
    logging.basicConfig(level=logging.CRITICAL, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logging.critical(f"CRITICAL IMPORT ERROR in main_fusion_public.py: {e}. ZOTHEOS may not function.", exc_info=True)
    # Provide fallbacks for critical components if possible, or allow failure
    # For now, if these fail, the __init__ will likely raise an error or log critical status.
    # Making the application fail loudly is often better than silent dysfunction.
    # Consider exiting if critical imports fail: sys.exit(f"Fatal Import Error in main_fusion_public.py: {e}")

# --- Start of FIX: Define LLAMA_CPP_AVAILABLE ---
try:
    # Attempt to import the core Llama class from llama_cpp
    from llama_cpp import Llama # You might also need LlamaCppError if you use it
    LLAMA_CPP_AVAILABLE = True
    # print("DEBUG: llama_cpp imported successfully, LLAMA_CPP_AVAILABLE=True") # Optional debug print
except ImportError:
    LLAMA_CPP_AVAILABLE = False
    # print("DEBUG: llama_cpp import failed, LLAMA_CPP_AVAILABLE=False") # Optional debug print
except Exception as e_llama_import: # Catch other potential errors during import
    LLAMA_CPP_AVAILABLE = False
    # print(f"DEBUG: An unexpected error occurred during llama_cpp import: {e_llama_import}, LLAMA_CPP_AVAILABLE=False") # Optional debug print
# --- End of FIX ---

logger = logging.getLogger("ZOTHEOS_MainFusion")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s] - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class MainFusionPublic:
    def __init__(self, device_preference: Optional[str] = "cuda"):
        logger.info("🚀 ZOTHEOS MainFusion Initializing (Tier Logic, Async Fusion Summary Enabled)...")

        self.config = {
            "MODEL_ROLES": MODEL_ROLES,
            "MODEL_ROLE_SYSTEM_PROMPTS": MODEL_ROLE_SYSTEM_PROMPTS,
            "DEFAULT_SYSTEM_PROMPT": DEFAULT_SYSTEM_PROMPT,
            "TIER_CONFIG": {
                "free": {"model_limit": 2, "memory_enabled": False, "display_name": "Free Tier"},
                "starter": {"model_limit": 3, "memory_enabled": True, "display_name": "Starter Tier"},
                # Pro uses max models from MODEL_PATHS, ensuring it uses all available configured models
                "pro": {"model_limit": len(MODEL_PATHS.keys()) if MODEL_PATHS else 3, "memory_enabled": True, "display_name": "Pro Tier"}
            }
        }
        self.models_last_queried_for_perspectives: List[str] = []

        try:
            # Pass global config values for ModelManager initialization
            logger.info(f"Initializing ModelManager with device_preference='{device_preference}', max_count={MAX_CONCURRENT_MODELS}, max_ram_gb={MAX_RAM_MODELS_GB}...")
            self.model_manager = ModelManager(
                device_preference=device_preference,
                max_model_count=MAX_CONCURRENT_MODELS,
                max_ram_models_gb=MAX_RAM_MODELS_GB
            )
            # active_model_names_in_order is the master list of all models ZOTHEOS *could* use
            self.active_model_names_in_order: List[str] = list(MODEL_PATHS.keys()) if MODEL_PATHS else []
            logger.info(f"✅ MainFusion initialized. Max available models for fusion: {self.active_model_names_in_order}")
        except Exception as e_mm_init:
            logger.critical(f"❌ CRITICAL: ModelManager failed to initialize in MainFusion: {e_mm_init}", exc_info=True)
            self.model_manager = None
            self.active_model_names_in_order = []

        try:
            self.memory_bank = MemoryBank()
            logger.info("✅ MemoryBank initialized.")
        except Exception as e_mb_init:
            logger.error(f"❌ MemoryBank failed to initialize: {e_mb_init}. Interactions may not be stored.", exc_info=True)
            self.memory_bank = None

        if not self.model_manager or not LLAMA_CPP_AVAILABLE: # Check if ModelManager itself or Llama backend failed
             logger.critical("MainFusion started in a DEGRADED state: ModelManager or Llama.cpp backend is UNAVAILABLE.")


    async def _get_single_model_response_direct(self, model_name: str, user_query: str, system_prompt_for_call: str, preset_name_for_call: str) -> Dict[str, Any]:
        response_text = f"[Error: Model '{model_name}' generation did not complete or model unavailable]"
        start_time_model = time.perf_counter()
        status = "Model Error or Unavailable"

        if not self.model_manager:
            logger.error(f"[{model_name}] ModelManager not available for generation.")
            return {"model": model_name, "text": "[Error: ModelManager is offline]", "time_ms": 0, "status": "ModelManager Offline"}

        try:
            logger.info(f"⚙️ [{model_name}] Calling ModelManager.generate_with_model. System Prompt: '{system_prompt_for_call[:50]}...'. User Query: '{user_query[:50].replace(chr(10),' ')}...'")
            # generate_with_model handles actual call to Llama instance via async_generation_wrapper
            response_text = await self.model_manager.generate_with_model(
                model_name=model_name,
                prompt=user_query,
                preset_name=preset_name_for_call,
                system_prompt=system_prompt_for_call
            )
            preview = response_text[:100].replace("\n", " ")
            if response_text.startswith(("[Error:", "[No text generated", "[Malformed response")) or not response_text.strip():
                logger.warning(f"⚠️ [{model_name}] Model returned an error string or empty content: {preview}...")
                status = "Model Error or Empty Response"
            else:
                logger.info(f"✅ [{model_name}] Response received ({len(response_text)} chars): {preview}...")
                status = "Success"
        except Exception as e:
            logger.error(f"❌ [{model_name}] Exception in _get_single_model_response_direct: {e}", exc_info=True)
            response_text = f"[Error: MainFusion encountered an exception during {model_name} generation: {type(e).__name__}]"
            status = "MainFusion Exception"
        inference_time_ms = (time.perf_counter() - start_time_model) * 1000
        return {"model": model_name, "text": response_text.strip(), "time_ms": round(inference_time_ms, 2), "status": status}

    async def _generate_true_fusion_summary(self, raw_responses_dict: Dict[str, str], original_query: str, models_actually_used: List[str]) -> str:
        logger.info(f"Attempting to generate a true fusion summary from {len(models_actually_used)} perspective(s).")
        summarizer_model_key = "gemma" # Default summarizer, ensure it's in MODEL_PATHS and loadable

        valid_responses_for_summary = {
            model: text for model, text in raw_responses_dict.items()
            if model in models_actually_used and text and not text.startswith("[Error:")
        }

        if not valid_responses_for_summary: # No valid responses at all
            logger.warning("No valid responses available to generate a fusion summary.")
            return "A summary could not be generated as no valid perspectives were successfully gathered."

        if len(valid_responses_for_summary) == 1:
            single_model_name = list(valid_responses_for_summary.keys())[0]
            logger.info(f"Only one valid perspective from {single_model_name}. Using it as the summary.")
            return f"Based on the single available perspective from **{single_model_name.capitalize()}**:\n\n{list(valid_responses_for_summary.values())[0]}"

        fusion_prompt = f"Original User Question: \"{original_query}\"\n\n"
        fusion_prompt += "You are the Fusion Summary engine for ZOTHEOS. Your task is to read the multiple perspectives provided below and write a concise, balanced, and synthesized summary. Capture the most important points, common themes, and notable differences from all viewpoints. Your tone should be thoughtful, neutral, and respectful. Structure the summary clearly.\n\n"

        for model_name, text in valid_responses_for_summary.items():
            role = self.config["MODEL_ROLES"].get(model_name, "General")
            fusion_prompt += f"--- PERSPECTIVE FROM {model_name.upper()} ({role.capitalize()}) ---\n{text}\n\n"

        fusion_prompt += "--- SYNTHESIZED SUMMARY (Combine the perspectives above into a unified insight) ---\n"

        logger.info(f"Calling summarizer model '{summarizer_model_key}' (from available: {self.active_model_names_in_order}) for fusion summary.")

        if not self.model_manager:
            logger.error("ModelManager not available for generating fusion summary.")
            return "[Error: Summarizer service unavailable - ModelManager offline]"

        try:
            # The system prompt for the summarizer itself
            summarizer_system_prompt = "You are an expert synthesis AI. Your role is to create a coherent and insightful summary from the provided texts."

            summary_text = await self.model_manager.generate_with_model(
                model_name=summarizer_model_key,
                prompt=fusion_prompt, # The full context with all perspectives
                preset_name="precise", # Use a precise preset for summarization
                system_prompt=summarizer_system_prompt
            )
            if summary_text and not summary_text.startswith("[Error:"):
                logger.info("✅ True fusion summary generated successfully.")
                return summary_text.strip()
            else:
                logger.warning(f"Summarizer model '{summarizer_model_key}' returned an error or empty response: {summary_text}")
                return "[Warning: Summary generation was partial or failed. Displaying raw perspectives.]"
        except Exception as e:
            logger.error(f"❌ Exception while generating true fusion summary with '{summarizer_model_key}': {e}", exc_info=True)
            return f"[Error: The summary generation process failed. Exception: {type(e).__name__}]"

    def _analyze_responses_basic(self, responses_dict: Dict[str, str], model_roles: Dict[str, str]) -> Dict[str, Any]:
        valid_responses = {model: text for model, text in responses_dict.items() if text and not text.startswith("[Error:")}
        consensus_points = []
        if len(valid_responses) > 1: consensus_points.append("Multiple perspectives were gathered and synthesized.")
        elif len(valid_responses) == 1: consensus_points.append("A single primary perspective was available for synthesis.")
        else: consensus_points.append("No valid primary perspectives were available for synthesis.")
        return {"consensus_points": consensus_points, "contradictions": [], "unique_insights": valid_responses}

    def _synthesize_fusion_response(self, analysis_result: dict, model_roles: dict, raw_responses_dict: dict, final_summary_text: str, models_used_for_perspectives: List[str]) -> str:
        response_parts = []

        response_parts.append("## ✨ ZOTHEOS Final Synthesized Insight ✨")
        response_parts.append(final_summary_text if final_summary_text and not final_summary_text.startswith(("[Error:", "[Warning:")) else "*Synthesis process encountered an issue or no summary was generated. Please see detailed perspectives below.*")
        response_parts.append("\n---\n")

        response_parts.append("### 💬 Detailed Individual Perspectives")
        has_any_valid_perspectives = False
        for model_name in models_used_for_perspectives:
            text = raw_responses_dict.get(model_name)
            role = model_roles.get(model_name, "General") # Default role
            response_parts.append(f"**Perspective from {model_name.capitalize()} ({role.capitalize()}):**")
            if text and not text.startswith("[Error:"):
                response_parts.append(text.strip())
                has_any_valid_perspectives = True
            else:
                response_parts.append(f"*{text if text else '[No response or error from this model.]'}*") # Display error or placeholder
            response_parts.append("")
        if not has_any_valid_perspectives and not (final_summary_text and not final_summary_text.startswith(("[Error:", "[Warning:"))):
             # If summary also failed/empty and no valid individual perspectives.
             response_parts = ["## ⚠️ ZOTHEOS Alert\n\nUnfortunately, ZOTHEOS encountered issues processing your query with all available AI cores for your tier. No insights could be gathered at this time. Please try rephrasing your query or try again later."]
        elif not has_any_valid_perspectives : # Summary might be there, but no individual details.
            response_parts.append("*No valid individual perspectives were successfully retrieved to display in detail.*")


        return "\n".join(response_parts).strip()

    async def process_query_with_fusion(
        self,
        query: str,
        user_token: Optional[str] = None,
        persona_key: Optional[str] = None,
        fusion_mode_override: str = "balanced",
        **kwargs
    ) -> str:
        process_start_time = time.time()
        current_tier_name = get_user_tier(user_token if user_token else "")
        tier_settings = self.config["TIER_CONFIG"].get(current_tier_name, self.config["TIER_CONFIG"]["free"])
        tier_model_limit = tier_settings["model_limit"]
        tier_memory_enabled = tier_settings["memory_enabled"]
        logger.info(f"User Tier: '{current_tier_name}' ({tier_settings['display_name']}). Model Limit: {tier_model_limit}, Memory: {'Enabled' if tier_memory_enabled else 'Disabled'}.")

        if not self.model_manager or not LLAMA_CPP_AVAILABLE: return "[Error: ZOTHEOS Core Model Manager not ready or Llama.cpp backend unavailable.]"
        if not self.active_model_names_in_order: return "[Error: ZOTHEOS Core not ready. No models configured in MODEL_PATHS.]"
        if not query or not query.strip(): return "[Error: Query is empty. Please provide a question or topic.]"

        current_query_text = query
        current_preset_name = fusion_mode_override if fusion_mode_override in INFERENCE_PRESETS else DEFAULT_INFERENCE_PRESET
        base_persona_prompt = SYSTEM_PERSONAS.get(persona_key or "default", self.config["DEFAULT_SYSTEM_PROMPT"])

        # Determine actual models to use based on tier limit and availability
        models_to_use_for_perspectives = [m for m in self.active_model_names_in_order if m in MODEL_PATHS][:tier_model_limit]
        self.models_last_queried_for_perspectives = models_to_use_for_perspectives # For status report

        if not models_to_use_for_perspectives:
            logger.error(f"No models available for tier '{current_tier_name}' after applying limit of {tier_model_limit}.")
            return f"[Error: No models available for your current tier ('{current_tier_name}').]"

        logger.info(f"🔎 Processing query. Models for perspectives (Tier: {current_tier_name}): {models_to_use_for_perspectives}. Preset: '{current_preset_name}'. Query: '{current_query_text[:60].replace(chr(10),' ')}...'")

        raw_responses_dict: Dict[str, str] = {}
        individual_results_for_memory: List[Dict[str, Any]] = []
        successful_responses = 0

        for model_name in models_to_use_for_perspectives:
            model_role = self.config["MODEL_ROLES"].get(model_name, "general")
            system_prompt_for_model = self.config["MODEL_ROLE_SYSTEM_PROMPTS"].get(model_role, base_persona_prompt)
            query_for_this_model = current_query_text
            if model_name.lower() == "gemma" and system_prompt_for_model:
                query_for_this_model = f"<start_of_turn>user\n{system_prompt_for_model.strip()}\n{current_query_text}<end_of_turn>\n<start_of_turn>model\n"
                system_prompt_for_model = ""
            model_output_data = await self._get_single_model_response_direct(model_name, query_for_this_model, system_prompt_for_model, current_preset_name)
            individual_results_for_memory.append(model_output_data)
            raw_responses_dict[model_name] = model_output_data.get("text", "[Error: No text field in response data]")
            if model_output_data.get("status") == "Success":
                successful_responses += 1

        synthesized_summary_text = await self._generate_true_fusion_summary(raw_responses_dict, current_query_text, models_to_use_for_perspectives)
        analysis_result = self._analyze_responses_basic(raw_responses_dict, self.config["MODEL_ROLES"]) # Basic analysis for now
        final_fused_output_content = self._synthesize_fusion_response(analysis_result, self.config["MODEL_ROLES"], raw_responses_dict, synthesized_summary_text, models_to_use_for_perspectives)

        persona_display = (persona_key or "default").capitalize()
        mode_display = current_preset_name.capitalize()
        tier_display_name = tier_settings.get("display_name", current_tier_name.capitalize())
        final_header = f"## 🧠 ZOTHEOS Fused Perspectives 🧠\n*(Fusion Mode: {mode_display} | Persona: {persona_display} | Tier: {tier_display_name})*\n\n"
        final_fused_output = final_header + final_fused_output_content

        if successful_responses == 0 and not "[Error:" in final_fused_output_content and not "[Warning:" in final_fused_output_content:
             logger.error(f"All models ({len(models_to_use_for_perspectives)}) failed for tier '{current_tier_name}'.")
             final_fused_output = final_header + "[Critical Error: ZOTHEOS was unable to obtain any valid responses from its AI cores for this query.]\n\n" + final_fused_output_content.split("\n\n",1)[-1]

        if tier_memory_enabled:
            if self.memory_bank:
                try:
                    memory_metadata = {
                        "user_token_used_prefix": user_token[:3] + "***" if user_token and len(user_token) > 3 else "N/A (No Token)" if not user_token else user_token,
                        "tier_at_interaction": current_tier_name,
                        "persona_key": persona_key or "default", "fusion_mode_used": current_preset_name,
                        # timestamp_iso is now added within store_memory_async
                        "duration_seconds": round(time.time() - process_start_time, 3),
                        "active_models_queried": models_to_use_for_perspectives, # Models actually used for perspectives
                        "individual_model_outputs": individual_results_for_memory, # Detailed dicts
                        "synthesized_summary_text": synthesized_summary_text, # The AI-generated summary
                        "fused_response_length_chars": len(final_fused_output),
                        "successful_model_responses": successful_responses,
                        "total_models_queried": len(models_to_use_for_perspectives)
                    }
                    await self.memory_bank.store_memory_async(query=current_query_text, response=final_fused_output, metadata=memory_metadata)
                except Exception as e_mem: logger.error(f"Failed to store fusion interaction in MemoryBank (Tier: '{current_tier_name}'): {e_mem}", exc_info=True)
            else: logger.warning(f"MemoryBank not initialized. Skipping storage (Tier: '{current_tier_name}').")
        else: logger.info(f"Memory storage disabled for tier '{current_tier_name}'. Skipping storage.")

        total_processing_time = round(time.time() - process_start_time, 2)
        logger.info(f"🧠 Fusion complete in {total_processing_time}s. Output len: {len(final_fused_output)}. Models used: {len(models_to_use_for_perspectives)} (Tier: {current_tier_name}).")
        return final_fused_output

    async def get_status_report(self) -> Dict[str, Any]:
        report: Dict[str, Any] = {
            "status": "Full Multi-Model Fusion Mode Active",
            "fusion_engine_status": "Online" if self.model_manager and self.active_model_names_in_order and LLAMA_CPP_AVAILABLE else "Degraded/Offline",
            "all_available_models": self.active_model_names_in_order,
            "models_last_queried_for_perspectives": getattr(self, 'models_last_queried_for_perspectives', []),
            "model_manager_status": "Online" if self.model_manager else "Offline/Init Failed",
            "llama_cpp_backend_available": LLAMA_CPP_AVAILABLE,
            "memory_bank_status": "Online" if self.memory_bank else "Offline/Init Failed"
        }
        if self.model_manager and hasattr(self.model_manager, 'get_loaded_model_stats'):
            try: report["model_manager_runtime_stats"] = self.model_manager.get_loaded_model_stats()
            except Exception as e: report["model_manager_runtime_stats"] = f"Error getting MM stats: {e}"
        else: report["model_manager_runtime_stats"] = "ModelManager N/A for stats."

        if self.memory_bank and hasattr(self.memory_bank, 'get_memory_stats'):
            try: report["memory_bank_stats"] = await self.memory_bank.get_memory_stats()
            except Exception as e: report["memory_bank_stats"] = f"Error getting MB stats: {e}"
        else: report["memory_bank_stats"] = "MemoryBank N/A for stats."
        return report

if __name__ == "__main__":
    if os.getenv("ZOTHEOS_DEBUG_DEPS", "false").lower() != "true":
        for lib_logger_name in ["torch", "huggingface_hub", "psutil", "llama_cpp", "httpx", "PIL"]: logging.getLogger(lib_logger_name).setLevel(logging.WARNING)
    logger.setLevel(logging.DEBUG)
    logger.info("--- MainFusionPublic (Tier Logic & Async Summary) CLI Test ---")
    async def run_main_fusion_cli_test_with_token(test_token=None, token_desc="Default (Free Tier)"):
        main_fusion_instance: Optional[MainFusionPublic] = None
        logger.info(f"\n--- Testing with token: '{token_desc}' ---")
        try:
            main_fusion_instance = MainFusionPublic(device_preference="cuda")
            if not main_fusion_instance.model_manager or not main_fusion_instance.active_model_names_in_order or not LLAMA_CPP_AVAILABLE:
                logger.critical("CLI Test Aborted: MainFusion init failed (MM or LlamaCPP unavailable)."); return
            test_query = "What are the core principles of Stoicism and how can they be applied in modern life?"
            logger.info(f"CLI Test: Querying (Token: {test_token[:3] + '...' if test_token and len(test_token)>3 else test_token}): '{test_query}'")
            response = await main_fusion_instance.process_query_with_fusion(query=test_query, user_token=test_token, persona_key="philosopher", fusion_mode_override="balanced")
            print("\n" + "="*25 + f" CLI Test Response ({token_desc}) " + "="*25); print(response); print("="* (50 + len(f" CLI Test Response ({token_desc}) ") + 2))
            status = await main_fusion_instance.get_status_report()
            print("\nSystem Status Report After Query:"); print(json.dumps(status, indent=2, default=str))
        except Exception as e: logger.critical(f"Error during CLI test ({token_desc}): {e}", exc_info=True); print(f"🚨 CLI Test Error ({token_desc}): {e}")
        finally:
            if main_fusion_instance and main_fusion_instance.model_manager and hasattr(main_fusion_instance.model_manager, 'shutdown'):
                logger.info(f"CLI Test ({token_desc}): Shutting down ModelManager...")
                main_fusion_instance.model_manager.shutdown() # This is synchronous
            logger.info(f"🛑 MainFusion CLI test ({token_desc}) shutdown.")
    async def run_all_cli_tests():
        # Ensure stripe_users.json is in project root for this test to work with tokens
        tokens_to_test = {
            None: "No Token (Defaults to Free)",
            "TOKEN_FOR_FREE_TEST": "Free Tier Token", # Add this to your stripe_users.json
            "TOKEN_FOR_STARTER_TEST": "Starter Tier Token", # Add this
            "TOKEN_FOR_PRO_TEST": "Pro Tier Token" # Add this
        }
        # Create dummy stripe_users.json if not exists for test
        if not os.path.exists(os.path.join(project_root, "stripe_users.json")):
            logger.warning("Creating dummy stripe_users.json for CLI test.")
            dummy_users = {t: t.split('_')[1].lower() for t in tokens_to_test if t} # type: ignore
            with open(os.path.join(project_root, "stripe_users.json"), "w") as f:
                json.dump(dummy_users, f, indent=2)

        for token, desc in tokens_to_test.items():
            await run_main_fusion_cli_test_with_token(token, desc)

    asyncio.run(run_all_cli_tests())