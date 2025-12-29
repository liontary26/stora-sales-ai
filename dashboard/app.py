import time
from datetime import datetime

import pandas as pd
import requests
import streamlit as st

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Store Sales Forecast",
    page_icon="📈",
    layout="wide",
)

# -----------------------------
# Session state
# -----------------------------
if "history" not in st.session_state:
    st.session_state.history = []  # list of dicts

if "api_health" not in st.session_state:
    st.session_state.api_health = None

# -----------------------------
# Helpers
# -----------------------------
def safe_get(url: str, timeout: int = 5):
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.json()

def safe_post(url: str, payload: dict, timeout: int = 10):
    r = requests.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()

def kpi_card(title: str, value: str, help_text: str | None = None):
    with st.container(border=True):
        st.caption(title)
        st.markdown(f"### {value}")
        if help_text:
            st.caption(help_text)

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.title("⚙️ Settings")

    api_url = st.text_input("API Base URL", value="http://127.0.0.1:8000").strip().rstrip("/")
    st.caption("Örn: http://127.0.0.1:8000")

    colA, colB = st.columns(2)
    with colA:
        btn_health = st.button("🔎 Health", use_container_width=True)
    with colB:
        btn_clear = st.button("🧹 Clear", use_container_width=True)

    if btn_clear:
        st.session_state.history = []
        st.session_state.api_health = None
        st.toast("Geçmiş temizlendi.", icon="✅")

    if btn_health:
        try:
            st.session_state.api_health = safe_get(f"{api_url}/health")
            st.success("API aktif ✅")
        except Exception as e:
            st.session_state.api_health = None
            st.error("API'ye ulaşılamadı ❌")
            st.code(str(e))

    st.divider()
    st.subheader("ℹ️ API Status")
    if st.session_state.api_health:
        st.json(st.session_state.api_health)
    else:
        st.info("Health için butona bas.")

# -----------------------------
# Header
# -----------------------------
st.title("📈 Store Sales Forecast Dashboard")
st.caption("FastAPI üzerinden satış tahmini (Store Sales Forecast API)")

# -----------------------------
# Layout: Input / Output
# -----------------------------
left, right = st.columns([1.2, 1])

with left:
    st.subheader("🧾 Tahmin Girdileri")

    with st.form("predict_form", border=True):
        c1, c2 = st.columns(2)

        with c1:
            onpromotion = st.number_input("onpromotion", min_value=0.0, value=10.0, step=1.0)
            is_holiday = st.selectbox("is_holiday (0/1)", options=[0, 1], index=0)
            lag_1 = st.number_input("lag_1 (dün satış)", min_value=0.0, value=200.0, step=1.0)

        with c2:
            lag_7 = st.number_input("lag_7 (geçen hafta aynı gün)", min_value=0.0, value=180.0, step=1.0)
            rolling_7_mean = st.number_input("rolling_7_mean", min_value=0.0, value=190.0, step=1.0)
            rolling_14_mean = st.number_input("rolling_14_mean", min_value=0.0, value=185.0, step=1.0)

        submitted = st.form_submit_button("🚀 Tahmin Et")

    payload = {
        "onpromotion": float(onpromotion),
        "is_holiday": int(is_holiday),
        "lag_1": float(lag_1),
        "lag_7": float(lag_7),
        "rolling_7_mean": float(rolling_7_mean),
        "rolling_14_mean": float(rolling_14_mean),
    }

    with st.expander("📤 Gönderilecek JSON", expanded=False):
        st.json(payload)

with right:
    st.subheader("📌 Sonuç")

    # Placeholders for KPI cards
    pred_placeholder = st.empty()
    meta_placeholder = st.empty()

    # If we have history, show last prediction
    if st.session_state.history:
        last = st.session_state.history[-1]
        pred_placeholder.container()
        kpi_card("Tahmini Satış", f"{last['prediction_sales']:,.2f}")
        meta_placeholder.container()
        kpi_card("Zaman", last["timestamp"], "Son tahminin alındığı zaman")
    else:
        with pred_placeholder.container():
            kpi_card("Tahmini Satış", "—", "Henüz tahmin alınmadı.")
        with meta_placeholder.container():
            kpi_card("Zaman", "—")

# -----------------------------
# Run prediction
# -----------------------------
if submitted:
    with st.spinner("Tahmin alınıyor..."):
        try:
            t0 = time.time()
            resp = safe_post(f"{api_url}/predict", payload)
            dt_ms = int((time.time() - t0) * 1000)

            pred = resp.get("prediction_sales", None)
            if pred is None:
                st.error("API yanıtında 'prediction_sales' bulunamadı.")
                st.json(resp)
            else:
                row = {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    **payload,
                    "prediction_sales": float(pred),
                    "latency_ms": dt_ms,
                }
                st.session_state.history.append(row)
                st.toast("Tahmin alındı ✅", icon="✅")

                # Update KPI cards immediately
                with right:
                    pred_placeholder.container()
                    kpi_card("Tahmini Satış", f"{float(pred):,.2f}")
                    meta_placeholder.container()
                    kpi_card("Gecikme", f"{dt_ms} ms", "API yanıt süresi")

        except requests.exceptions.HTTPError as e:
            st.error("❌ API hata döndürdü (HTTPError).")
            st.code(str(e))
        except Exception as e:
            st.error("❌ İstek başarısız oldu.")
            st.code(str(e))

# -----------------------------
# History table
# -----------------------------
st.divider()
st.subheader("🗂️ Tahmin Geçmişi")

if st.session_state.history:
    df = pd.DataFrame(st.session_state.history)
    st.dataframe(df, use_container_width=True, hide_index=True)

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ CSV indir",
        data=csv,
        file_name="prediction_history.csv",
        mime="text/csv",
    )
else:
    st.info("Henüz geçmiş yok. Üstten bir tahmin al.")

# -----------------------------
# Footer
# -----------------------------
st.caption("Not: API çalışmıyorsa önce terminalde `python -m uvicorn api.main:app --reload --port 8000` ile başlat.")
