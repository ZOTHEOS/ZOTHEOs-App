import asyncio
import logging
import torch
from difflib import SequenceMatcher
from time import time
from typing import Dict, List, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config_settings_public import MODEL_PATHS, MODEL_CONFIG, MODEL_WEIGHTS
from model_manager_public import ModelManager
from response_optimizer_public import ResponseOptimizer

logger = logging.getLogger(__name__)


class QueryProcessor:
    def __init__(self, model_manager: ModelManager, response_optimizer: ResponseOptimizer):
        self.model_manager = model_manager
        self.response_optimizer = response_optimizer
        self.model_paths = MODEL_PATHS
        self.model_config = MODEL_CONFIG
        self.memory_bank = []

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True
            torch.set_float32_matmul_precision('high')
            logger.info(f"✅ QueryProcessor initialized on {self.device}")
        else:
            logger.warning("⚠️ Running on CPU")

        self.model_tiers = {
            "balanced": ["gemma", "mistral", "llama"]
        }

    def select_models(self, query: str) -> List[str]:
        return [m for m in self.model_tiers["balanced"] if m in self.model_paths]

    async def process_query_with_fusion(self, query: str) -> str:
        selected_models = self.select_models(query)
        logger.info(f"🧠 Selected models: {selected_models}")

        tasks = [asyncio.create_task(self.invoke_model(m, query)) for m in selected_models]
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        valid_responses = {
            m: r for m, r in zip(selected_models, responses)
            if isinstance(r, str) and r.strip()
        }

        if not valid_responses:
            logger.error("❌ All models failed or returned empty responses.")
            return "⚠️ No valid responses received from models."

        logger.info(f"✅ {len(valid_responses)} valid responses received.")
        fused = await self.advanced_fusion(valid_responses, query)
        self.memory_bank.append({"query": query, "response": fused, "timestamp": time()})
        return fused

    async def invoke_model(self, model_name: str, query: str) -> str:
        try:
            model = self.model_manager.loaded_models.get(model_name)
            if not model:
                model = self.model_manager.load_model_from_disk(model_name)
                if not model:
                    raise RuntimeError(f"Model '{model_name}' failed to load.")

            config = self.model_config.get(model_name, {})
            max_tokens = config.get("max_tokens", 800)
            temperature = config.get("temperature", 0.7)

            logger.info(f"🔧 Invoking '{model_name}' with temp={temperature}, max_tokens={max_tokens}")

            with torch.no_grad():
                completion = await asyncio.to_thread(
                    model.create_completion,
                    prompt=query,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stop=["</s>", "User:", "Assistant:"]
                )

            response_text = completion.get("choices", [{}])[0].get("text", "").strip()
            return response_text

        except Exception as e:
            logger.error(f"❌ Error in model '{model_name}': {e}")
            return ""

    def clean_response(self, raw) -> str:
        if isinstance(raw, str):
            return raw.strip()

        if isinstance(raw, dict):
            return raw.get("generated_text", "").strip() or raw.get("text", "").strip()

        if isinstance(raw, list):
            if isinstance(raw[0], dict) and "generated_text" in raw[0]:
                return raw[0]["generated_text"].strip()
            return " ".join(str(x).strip() for x in raw if isinstance(x, str))

        return str(raw).strip()

    async def advanced_fusion(self, responses: Dict[str, str], query: str) -> str:
        cleaned = {m: self.clean_response(r) for m, r in responses.items()}
        seen = set()
        unique: List[Tuple[str, float, str]] = []

        for model, text in cleaned.items():
            if not text or any(self.is_similar(text, prev) for prev in seen):
                continue
            seen.add(text)
            score = self.score_response(text, query)
            unique.append((text, score, model))

        if not unique:
            return "⚠️ No high-quality content to merge."

        unique.sort(key=lambda x: x[1], reverse=True)
        best_responses = [await self.response_optimizer.optimize_response(t, query) for t, _, _ in unique[:3]]
        return "\n---\n".join(best_responses).strip()

    def is_similar(self, a: str, b: str, threshold: float = 0.9) -> bool:
        return SequenceMatcher(None, a.strip().lower(), b.strip().lower()).ratio() > threshold

    def score_response(self, response: str, query: str) -> float:
        try:
            tfidf = TfidfVectorizer().fit_transform([response, query])
            return cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
        except:
            return 0.5
