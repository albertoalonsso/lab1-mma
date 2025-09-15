from typing import Dict, Any
from src.orchestrator.multimodel_analyzer import MultiModelAnalyzer
from src.pipelines.rag_index import retrieve_context

class FinancialNewsAnalyzer:
    def __init__(self):
        self.mm = MultiModelAnalyzer()

    def _rule_based(self, text: str):
        t = text.lower()
        pos = ["beat","beats","growth","record","surge","bullish","upgrade","profit"]
        neg = ["miss","misses","downgrade","loss","decline","bearish","probe","fraud"]
        score = sum(w in t for w in pos) - sum(w in t for w in neg)
        s = "neutral"
        if score > 0: s = "bullish"
        if score < 0: s = "bearish"
        return {"sentiment": s, "score": score}

    async def analyze_sentiment(self, text: str, provider: str = "stub", use_rag: bool = False) -> Dict[str, Any]:
        q = text
        ctx_used = []
        if use_rag:
            ctx = retrieve_context(text)
            if ctx:
                ctx_used = ctx
                ctx_blob = "\n---\n".join(ctx[:3])
                q = f"CONTEXT:\n{ctx_blob}\n\nTEXT:\n{text}"
        rb = self._rule_based(text)
        routed = await self.mm.analyze_with_routing(q, task_type="news", provider=provider)
        return {"rule_based": rb, "provider": provider, "model_result": routed}
