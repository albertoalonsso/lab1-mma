import os, asyncio, json, re, time
from typing import Dict, Any, Optional
from tenacity import retry, wait_exponential, stop_after_attempt
from openai import OpenAI

from src.benchmarking.pricing import price_for

class DeepkSeekClient:
    def __init__(self, model: Optional[str] = None):
        self.api_key = os.getenv("DEEPSEEK_API_KEY")
        self.model = model or os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
        self.client = OpenAI(api_key=self.api_key, base_url = "https://api.deepseek.com") if self.api_key else None
        self.price = price_for("deepseek", 
                               self.model, 
                               cache_hit=(os.getenv("DEEPSEEK_CACHE_HIT","false").lower() in {"1","true","yes"}))

    def _user_msg(self, text: str):
        system = (
            "You are a financial NLP assistant. "
            "Respond ONLY with a valid JSON object using these fields: "
            "sentiment (bullish|bearish|neutral), confidence (0..1), "
            "key_entities (array of strings), impact_score (0..1)."
        )
        user = f"Analyze the following financial headline or news text:\n---\n{text}\n---"
        return [{"role":"system","content":system},{"role":"user","content":user}]

    @retry(wait=wait_exponential(min=1, max=10), stop=stop_after_attempt(4))
    def _call(self, messages):
        start = time.perf_counter()
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0,
            max_tokens = 200,
            response_format = {"type": "json_object"}
        )
        latency_ms = (time.perf_counter()-start)*1000
        return resp, latency_ms

    async def analyze(self, prompt: str) -> Dict[str, Any]:
        if not self.api_key:
            return {"ok": False, "provider":"deepseek", "error":"DEEPSEEK_API_KEY missing"}
        try:
            resp, latency_ms = await asyncio.to_thread(self._call, self._user_msg(prompt))
            text = resp.choices[0].message.content or ""
            parsed = None
            try:
                parsed = json.loads(text)
            except Exception:
                m = re.search(r"\{.*\}", text, re.S)
                if m:
                    parsed = json.loads(m.group(0))
            u = getattr(resp, "usage", None)
            usage = {
                "prompt": getattr(u, "prompt_tokens", None),
                "completion": getattr(u, "completion_tokens", None),
                "total": getattr(u, "total_tokens", None),
            } if u else {}
            cost = None
            if self.price and usage.get("prompt") is not None and usage.get("completion") is not None:
                cost = usage["prompt"]*self.price["in"] + usage["completion"] * self.price["out"]

            return {
                "ok": True, 
                "provider": "deepseek", 
                "model": self.model,
                "raw_text": text, 
                "parsed": parsed,
                "usage": usage,
                "latency_ms": round(latency_ms, 1),
                "cost_usd": round(cost, 6) if cost is not None else None,
            }
        
        except Exception as e:
            return {"ok": False, "provider": "deepseek", "model": self.model, "error": str(e), "error_type": type(e).__name__}