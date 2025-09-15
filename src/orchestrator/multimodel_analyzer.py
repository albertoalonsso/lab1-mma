import asyncio
from typing import Dict, Any, List
from dotenv import load_dotenv
from src.clients.base import env_keys_status
from src.clients.openai_client import OpenAIClient
from src.clients.deepseek_client import DeepkSeekClient
from src.clients.anthropic_client import AnthropicClient

class MultiModelAnalyzer:
    def __init__(self):
        load_dotenv(".env", override=True)
        self.keys = env_keys_status()
        self.openai = OpenAIClient()
        self.deepseek = DeepkSeekClient()
        self.anthropic = AnthropicClient()
    
    def _available(self) -> List[str]:
        avail = []
        if getattr(self.openai, "api_key", None):     avail.append("openai")
        if getattr(self.anthropic, "api_key", None):  avail.append("anthropic")
        if getattr(self.deepseek, "api_key", None):   avail.append("deepseek")
        return avail
    

    async def analyze_with_routing(self, query: str, task_type: str = "news", provider: str = "stub") -> Dict[str, Any]:
        if provider == "auto":
            for cand in ["openai", "deepseek", "anthropic"]:
                if cand in self._available():
                    provider = cand
                    break
            else:
                provider = "stub"

        if provider == "openai":
            res = await self.openai.analyze(query)
        elif provider == "anthropic":
            res = await self.anthropic.analyze(query)
        elif provider == "deepseek":
            res = await self.deepseek.analyze(query)
        else:
            return {
                "router_decision": "stub",
                "response": {"ok": True, "provider": "stub", "msg": "no external calls"},
                "keys_status": env_keys_status(),
                "task_type": task_type,
            }

        return {
            "router_decision": provider,
            "response": res,
            "keys_status": env_keys_status(),
            "task_type": task_type,
        }
    
    async def analyze_all_providers(self, query: str) -> Dict[str, Any]:
        tasks, providers = [], []
        if getattr(self.openai, "api_key", None):
            tasks.append(self.openai.analyze(query)); providers.append("openai")
        if getattr(self.anthropic, "api_key", None):
            tasks.append(self.anthropic.analyze(query)); providers.append("anthropic")
        if getattr(self.deepseek, "api_key", None):
            tasks.append(self.deepseek.analyze(query)); providers.append("deepseek")

        if not tasks:
            return {"available": [], "results": {}, "keys_status": env_keys_status()}

        results = await asyncio.gather(*tasks, return_exceptions=False)
        return {
            "available": providers,
            "results": {p: r for p, r in zip(providers, results)},
            "keys_status": env_keys_status()
        }

