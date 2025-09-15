from typing import Dict, List, Optional, Tuple
from src.benchmarking.pricing import price_for

def estimate_tokens (text: str) -> int:
    # ~ 1.3 token per word
    return max(1, int(len(text.split())*1.3))

def estimate_cost (provider:str, model: str, in_tokens: int, out_tokens: int, *, cache_hit=False) -> Optional[float]:
    p = price_for(provider, model, cache_hit=cache_hit)
    if not p: return None
    return in_tokens * p["in"] + out_tokens * p["out"]

def choose_provider (
        text: str,
        available: List[Tuple[str, str]],
        *,
        target: str="cheapest",
        expected_out_tokens: int = 120,
        sensitive: bool = False,
        budget_per_call_usd: Optional[float] = None,
) -> Dict:
    if sensitive:
        for prov, model in available:
            if prov == "anthropic":
                in_tok = estimate_tokens(text)
                cost = estimate_cost(prov, model, in_tok, expected_out_tokens) or 0.0
                return {"provider": prov, "model": model, "reason": "sensitive->anthropic", "est_cost_usd": round(cost, 6)}
    
    in_tok = estimate_tokens(text)
    scores = []
    for prov, model in available:
        cost = estimate_cost(prov, model, in_tok, expected_out_tokens) or 1e9
        if budget_per_call_usd is not None and cost > budget_per_call_usd:
            continue
        lat_bonus = 0.0
        if prov == "openai": lat_bonus = -0.00003
        if prov == "anthropic": lat_bonus  =-0.00002
        scores.append((cost + lat_bonus, prov, model, cost))

    if not scores:
        return {"provider": "stub", "model": None, "reason": "no provider within budget", "est_cost_usd": 0.0}
    
    scores.sort(key=lambda x: x[0])
    best = scores[0]
    return {"provider": best[1], "model": best[2], "reason": f"{target}", "est_cost_usd": round(best[3], 6)}