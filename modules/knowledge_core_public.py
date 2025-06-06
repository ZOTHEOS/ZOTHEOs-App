import os
import json
import logging
import asyncio
import traceback
from typing import List

logger = logging.getLogger(__name__)

# ✅ Define Knowledge Storage Path
KNOWLEDGE_FILE = os.path.expanduser("~/zotheos_public/zotheos_knowledge.json")

class KnowledgeCore:
    """🚀 Public Version of Knowledge Core (Simplified Storage and Retrieval)"""

    def __init__(self):
        """✅ Initialize Knowledge Storage."""
        self.knowledge_store = {}
        self.load_knowledge()

    def load_knowledge(self):
        """✅ Load Knowledge from JSON."""
        try:
            if os.path.exists(KNOWLEDGE_FILE):
                with open(KNOWLEDGE_FILE, "r", encoding="utf-8") as file:
                    self.knowledge_store = json.load(file)
                    logger.info(f"✅ Knowledge loaded from {KNOWLEDGE_FILE}")
            else:
                self.knowledge_store = {}
                logger.info(f"✅ Created new knowledge store at {KNOWLEDGE_FILE}")
        except Exception as e:
            logger.warning(f"⚠️ Error loading knowledge: {e}")
            self.knowledge_store = {}

    def save_knowledge(self):
        """✅ Save Knowledge to JSON."""
        try:
            with open(KNOWLEDGE_FILE, "w", encoding="utf-8") as file:
                json.dump(self.knowledge_store, file, indent=4)
            logger.info(f"✅ Knowledge saved to {KNOWLEDGE_FILE}")
        except Exception as e:
            logger.warning(f"⚠️ Error saving knowledge: {e}")

    async def update(self, new_knowledge: str, category: str = "general") -> bool:
        """✅ Add New Knowledge to the Store."""
        try:
            knowledge_id = str(len(self.knowledge_store) + 1)
            self.knowledge_store[knowledge_id] = {
                'content': new_knowledge,
                'category': category
            }
            self.save_knowledge()
            logger.info(f"✅ Knowledge added with ID {knowledge_id}")
            return True
        except Exception as e:
            logger.error(f"❌ Error updating knowledge: {traceback.format_exc()}")
            return False

    async def retrieve(self, query: str, n_results: int = 5) -> List[dict]:
        """🔎 Retrieve Relevant Knowledge Based on Simple Keyword Match."""
        try:
            matches = []
            for knowledge_id, data in self.knowledge_store.items():
                if query.lower() in data['content'].lower():
                    matches.append({
                        'id': knowledge_id,
                        'content': data['content'],
                        'category': data['category']
                    })
                    if len(matches) >= n_results:
                        break
            logger.info(f"✅ Retrieved {len(matches)} matches for query '{query}'")
            return matches
        except Exception as e:
            logger.error(f"❌ Error retrieving knowledge: {traceback.format_exc()}")
            return []

    async def reset(self) -> bool:
        """🗑️ Reset Knowledge Store (Delete All Stored Data)."""
        try:
            self.knowledge_store = {}
            self.save_knowledge()
            logger.info("✅ Knowledge store reset successfully.")
            return True
        except Exception as e:
            logger.error(f"❌ Error resetting knowledge store: {traceback.format_exc()}")
            return False

    async def delete_knowledge(self, knowledge_id: str) -> bool:
        """🗑️ Delete Specific Knowledge Entry by ID."""
        try:
            if knowledge_id in self.knowledge_store:
                del self.knowledge_store[knowledge_id]
                self.save_knowledge()
                logger.info(f"✅ Deleted knowledge ID: {knowledge_id}")
                return True
            else:
                logger.warning(f"⚠️ Knowledge ID {knowledge_id} not found.")
                return False
        except Exception as e:
            logger.error(f"❌ Error deleting knowledge ID {knowledge_id}: {traceback.format_exc()}")
            return False
