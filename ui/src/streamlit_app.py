from __future__ import annotations

import streamlit as st

from pages.inference_page import render as render_inference
from pages.offline_gates_page import render as render_offline_gates
from pages.online_control_page import render as render_online_control
from pages.review_queue_page import render as render_review_queue
from pages.run_explorer_page import render as render_run_explorer
from ui_api import ApiClient


def main() -> None:
    st.set_page_config(page_title="LLM Eval Ops", layout="wide")

    api_base = st.sidebar.text_input("API base URL", value="http://127.0.0.1:8000")
    page = st.sidebar.radio("Page", ["Inference", "Run Explorer", "Offline Gates", "Review Queue", "Online Control"])

    api = ApiClient(api_base=api_base)

    if page == "Inference":
        render_inference(api)
    elif page == "Run Explorer":
        render_run_explorer(api)
    elif page == "Offline Gates":
        render_offline_gates(api)
    elif page == "Review Queue":
        render_review_queue(api)
    elif page == "Online Control":
        render_online_control(api)


main()

