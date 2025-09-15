OPENAI_PRICES = {
    "gpt-4o-mini": {"in": 0.60/1_000_000, "out": 2.40/1_000_000},
}

ANTHROPIC_PRICES = {
    "claude-3-5-haiku":  {"in": 0.80/1_000_000, "out": 4.00/1_000_000},
    "claude-3-5-sonnet": {"in": 3.00/1_000_000, "out": 15.00/1_000_000},
    "claude-3-7-sonnet": {"in": 3.00/1_000_000, "out": 15.00/1_000_000},
    "claude-sonnet-4":   {"in": 3.00/1_000_000, "out": 15.00/1_000_000},
}

DEEPSEEK_PRICES = {
    # DeepSeek distinguishes between ‘cache miss’ and ‘cache hit’ at the entry point.
    "deepseek-chat":     {"in_miss": 0.27/1_000_000, "in_hit": 0.07/1_000_000, "out": 1.10/1_000_000},
    "deepseek-reasoner": {"in_miss": 0.55/1_000_000,                           "out": 2.19/1_000_000},
}

def _match_price(table: dict, model: str):
    m = (model or "").lower()
    for key, price in table.items():
        if key in m:
            return price
    return None

def price_for(provider: str, model: str, *, cache_hit: bool=False):
    p = provider.lower()
    if p == "openai":
        return _match_price(OPENAI_PRICES, model) or OPENAI_PRICES["gpt-4o-mini"]
    if p == "anthropic":
        # por defecto, si no matchea, asumimos haiku (barato)
        return _match_price(ANTHROPIC_PRICES, model) or ANTHROPIC_PRICES["claude-3-5-haiku"]
    if p == "deepseek":
        d = _match_price(DEEPSEEK_PRICES, model) or DEEPSEEK_PRICES["deepseek-chat"]
        in_price = d.get("in_hit" if cache_hit and "in_hit" in d else "in_miss")
        return {"in": in_price, "out": d["out"]}
    return None