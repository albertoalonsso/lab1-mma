import os, asyncio, json, re, time
from typing import Dict, Any, Optional
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
import anthropic
from src.benchmarking.pricing import price_for

class AnthropicClient:
    def __init__(self, model: Optional[str] = None):
        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        self.model = model or os.getenv("ANTHROPIC_MODEL", "claude-3-5-haiku-latest")
        self.client = anthropic.Anthropic(api_key=self.api_key) if self.api_key else None
        self.price = price_for("anthropic", self.model)

    def _instr(self) -> str:
        return (
            "You are a financial NLP assistant. "
            "Respond ONLY with a valid JSON object using these fields: "
            "sentiment (bullish|bearish|neutral), confidence (0..1), "
            "key_entities (array of strings), impact_score (0..1)."
        )

    # Estilo 1: system + bloques (SDKs recientes)
    @retry(wait=wait_exponential(min=1, max=10), stop=stop_after_attempt(3),
           retry=retry_if_exception_type((TypeError, anthropic.RateLimitError)))
    def _call_style1(self, user_text: str):
        start = time.perf_counter()
        resp = self.client.messages.create(
            model=self.model,
            system=self._instr(),
            max_tokens=200,
            temperature=0,
            messages=[{"role":"user","content":[{"type":"text","text": user_text}]}],
        )
        return resp, (time.perf_counter()-start)*1000

    # Estilo 2: sin 'system', instrucciones inyectadas en el contenido (string)
    def _call_style2(self, user_text: str):
        start = time.perf_counter()
        merged = f"{self._instr()}\n\nAnalyze this financial text strictly as JSON:\n---\n{user_text}\n---"
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=200,
            temperature=0,
            messages=[{"role":"user","content": merged}],
        )
        return resp, (time.perf_counter()-start)*1000

    # Estilo 3: bloques pero sin 'system' (otra variante aceptada por SDK intermedios)
    def _call_style3(self, user_text: str):
        start = time.perf_counter()
        merged = f"{self._instr()}\n\nAnalyze this financial text strictly as JSON:\n---\n{user_text}\n---"
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=200,
            temperature=0,
            messages=[{"role":"user","content":[{"type":"text","text": merged}]}],
        )
        return resp, (time.perf_counter()-start)*1000

    async def analyze(self, prompt: str) -> Dict[str, Any]:
        if not self.api_key:
            return {"ok": False, "provider":"anthropic", "error":"ANTHROPIC_API_KEY missing"}
        try:
            # Intento 1
            try:
                resp, latency_ms = await asyncio.to_thread(self._call_style1, prompt)
            except TypeError:
                # Intento 2
                try:
                    resp, latency_ms = await asyncio.to_thread(self._call_style2, prompt)
                except Exception:
                    # Intento 3
                    resp, latency_ms = await asyncio.to_thread(self._call_style3, prompt)

            # Extraer texto de bloques (SDKs Claude 3.x)
            text = ""
            if hasattr(resp, "content"):
                for b in resp.content:
                    t = getattr(b, "text", None)
                    if t: text += t
            if not text and hasattr(resp, "message"):
                text = getattr(resp.message, "content", "") or ""

            # Parse robusto
            parsed = None
            try:
                parsed = json.loads(text)
            except Exception:
                m = re.search(r"\{.*\}", text, re.S)
                if m: parsed = json.loads(m.group(0))

            # Usage + coste
            u = getattr(resp, "usage", None)
            usage = {
                "prompt": getattr(u, "input_tokens", None),
                "completion": getattr(u, "output_tokens", None),
                "total": ((getattr(u, "input_tokens", 0) or 0) + (getattr(u, "output_tokens", 0) or 0)),
            } if u else {}

            cost = None
            if self.price and usage.get("prompt") is not None and usage.get("completion") is not None:
                cost = usage["prompt"]*self.price["in"] + usage["completion"]*self.price["out"]

            return {
                "ok": True,
                "provider": "anthropic",
                "model": self.model,
                "raw_text": text,
                "parsed": parsed,
                "usage": usage,
                "latency_ms": round(latency_ms, 1),
                "cost_usd": round(cost, 6) if cost is not None else None,
            }
        except Exception as e:
            return {"ok": False, "provider": "anthropic", "model": self.model, "error": str(e), "error_type": type(e).__name__}
