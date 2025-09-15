import asyncio, json, time
from typing import List
from src.orchestrator.multimodel_analyzer import MultiModelAnalyzer

SAMPLE = [
    "Acme Corp beats earnings expectations amid record growth",
    "Acme Corp misses earnings; executives warn of declining demand",
    "Acme Corp announces quarterly results and guidance",
]

async def run_once(texts: List[str], providers: List[str]):
    mm = MultiModelAnalyzer()
    out = []
    for t in texts:
        for p in providers:
            start = time.perf_counter()
            res = await mm.analyze_with_routing(t, provider=p)
            dt = (time.perf_counter()-start)*1000
            r = res["response"]
            out.append({
                "provider": p, "model": r.get("model"), "ok": r.get("ok"),
                "latency_ms": r.get("latency_ms", dt), "cost_usd": r.get("cost_usd"),
                "sentiment": (r.get("parsed") or {}).get("sentiment")
            })
    return out

def main():
    import argparse, pandas as pd
    ap = argparse.ArgumentParser()
    ap.add_argument("--providers", nargs="+", default=["openai","anthropic","deepseek"])
    ap.add_argument("--texts", nargs="*", default=SAMPLE)
    args = ap.parse_args()

    res = asyncio.run(run_once(args.texts, args.providers))
    df = pd.DataFrame(res)
    print("\n== Summary ==")
    print(df.groupby("provider")[["latency_ms","cost_usd"]].mean().round(3))
    df.to_csv("tests/quick_benchmark.csv", index=False)
    print("\nSaved: tests/quick_benchmark.csv")

if __name__ == "__main__":
    main()