import os, sys, asyncio
from dotenv import load_dotenv

# Asegura que el proyecto raíz esté en sys.path (funciona al ejecutar desde root)
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

load_dotenv(".env", override=True)

import streamlit as st
from src.clients.base import env_keys_status
from src.pipelines.news_analyzer import FinancialNewsAnalyzer

st.set_page_config(page_title="Lab1 — Multi-Model Analyzer", layout="centered")
st.title("Lab1 — Multi-Model Financial Analyzer (stub)")

st.subheader("Estado de claves")
st.json(env_keys_status())

text = st.text_area("Pega una noticia / titular financiero",
                    "Acme Corp beats earnings expectations amid record growth.")
if st.button("Analizar (stub)"):
    analyzer = FinancialNewsAnalyzer()
    res = asyncio.run(analyzer.analyze_sentiment(text))
    st.subheader("Resultado (stub)")
    st.json(res)

st.caption("Nota: Stub sin llamadas a proveedores; valida entorno/estructura.")
