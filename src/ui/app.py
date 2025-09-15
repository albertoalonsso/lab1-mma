import os, sys, asyncio
from dotenv import load_dotenv

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

load_dotenv(".env", override=True)

import streamlit as st
from src.clients.base import env_keys_status
from src.pipelines.news_analyzer import FinancialNewsAnalyzer
from src.orchestrator.multimodel_analyzer import MultiModelAnalyzer

st.set_page_config(page_title="Lab1 — Multi-Model Analyzer", layout="wide")
st.title("Lab1 — Multi-Model Financial Analyzer")
st.caption("UI v0.4 — OpenAI + Anthropic + DeepSeek + Compare/Ensemble")

st.sidebar.header("Estado de claves")
st.sidebar.json(env_keys_status())

text = st.text_area("Pega una noticia / titular financiero",
                    "Acme Corp beats earnings expectations amid record growth.")

mode = st.radio("Proveedor (single run)", ["stub (rule-based)", "openai", "anthropic", "deepseek", "auto", "cost-aware"],
                 index=1, horizontal=True)

provider = "stub" if mode.startswith("stub") else mode

col1, col2 = st.columns([1,1])
with col1:
    if st.button("Analizar (single)"):
        analyzer = FinancialNewsAnalyzer()
        res = asyncio.run(analyzer.analyze_sentiment(text, provider=provider))
        st.subheader("Resultado (single)")
        st.json(res)
with col2:
    if st.button("Comparar (3)"):
        mm = MultiModelAnalyzer()
        allres = asyncio.run(mm.analyze_all_providers(text))
        st.write("Available providers in compare:", allres.get("available"))
        st.write("Keys status:", allres.get("keys_status"))
        st.subheader("Resultados por proveedor")
        cols = st.columns(3)
        order = ["openai","anthropic","deepseek"]
        for i, prov in enumerate(order):
            if prov in allres.get("results", {}):
                r = allres["results"][prov]
                with cols[i]:
                    st.markdown(f"### {prov.title()}")
                    st.json(r)
                    if r.get("ok") and r.get("parsed"):
                        p = r["parsed"]; u = r.get("usage") or {}
                        st.metric("Sentiment", p.get("sentiment","-"))
                        st.metric("Confidence", p.get("confidence",0.0))
                        st.metric("Impact", p.get("impact_score",0.0))
                        st.metric("Tokens (prompt)", u.get("prompt"))
                        st.metric("Tokens (completion)", u.get("completion"))
                        st.metric("Latency (ms)", r.get("latency_ms"))
                        if r.get("cost_usd") is not None:
                            st.metric("Coste (USD)", f"{r['cost_usd']:.6f}")

        # Ensemble sencillo (voto ponderado por confidence)
        votes = {}
        for prov, r in allres.get("results", {}).items():
            if r.get("ok") and r.get("parsed"):
                p = r["parsed"]
                votes[prov] = (p.get("sentiment"), float(p.get("confidence", 0.0)))
        if votes:
            weights = {"openai":0.4, "anthropic":0.35, "deepseek":0.25}
            scores = {"bullish":0.0,"bearish":0.0,"neutral":0.0}
            for prov, (label, conf) in votes.items():
                w = weights.get(prov, 0.3)
                scores[label] += w * (0.5 + 0.5*conf)
            label = max(scores, key=scores.get)
            reliability = max(scores.values()) - sorted(scores.values())[-2]
            st.subheader("Ensemble")
            st.write({"label": label, "reliability": round(reliability,3), "scores": {k:round(v,3) for k,v in scores.items()}})
        else:
            st.info("No hay resultados válidos para calcular ensemble.")

        costs = {prov: r.get("cost_usd") for prov, r in allres.get("results", {}).items() if r.get("cost_usd") is not None}
        if costs:
            st.subheader("Cost comparison")
            sum_cost = sum(costs.values())
            cheapest = min(costs.items(), key=lambda x: x[1])
            most = max(costs.items(), key=lambda x: x[1])
            c1,c2,c3 = st.columns(3)
            c1.metric("Cheapest provider", f"{cheapest[0]}", f"${cheapest[1]:.6f}")
            c2.metric("Most expensive", f"{most[0]}", f"${most[1]:.6f}")
            c3.metric("Ensemble sum (USD)", f"{sum_cost:.6f}")