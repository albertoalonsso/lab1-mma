from typing import Dict, Any
from src.orchestrator.multimodel_analyzer import MultiModelAnalyzer

class FinancialNewsAnalyzer:
    def __init__(self):
        self.mm = MultiModelAnalyzer()

    async def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        t = text.lower()
        pos = ["beat","beats","growth","record","surge","bullish","upgrade","profit"]
        neg = ["miss","misses","downgrade","loss","decline","bearish","probe","fraud"]
        score = sum(w in t for w in pos) - sum(w in t for w in neg)
        sentiment = "neutral"
        if score > 0: sentiment = "bullish"
        if score < 0: sentiment = "bearish"
        routed = await self.mm.analyze_with_routing(text, task_type="news")
        return {"sentiment": sentiment, "rule_based_score": score, "routed_info": routed}
