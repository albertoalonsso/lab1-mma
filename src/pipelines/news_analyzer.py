from typing import Dict, Any
from src.orchestrator.multimodel_analyzer import MultiModelAnalyzer

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

    async def analyze_sentiment(self, text: str, provider: str = "stub") -> Dict[str, Any]:
        rb = self._rule_based(text)
        routed = await self.mm.analyze_with_routing(text, task_type="news", provider=provider)
        return {"rule_based": rb, "provider": provider, "model_result": routed}
