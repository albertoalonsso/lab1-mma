from typing import Dict, Any
from dotenv import load_dotenv
from src.clients.base import env_keys_status

class MultiModelAnalyzer:
    def __init__(self):
        load_dotenv(".env", override=True)
        self.keys = env_keys_status()

    async def analyze_with_routing(self, query: str, task_type: str = "news") -> Dict[str, Any]:
        available = [k for k,v in self.keys.items() if v == "set"]
        return {
            "ok": True,
            "task_type": task_type,
            "query": query,
            "available_providers": available,
            "keys_status": self.keys,
            "result": "stub-only (sin llamadas a modelos aún)"
        }
