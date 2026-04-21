from __future__ import annotations

import streamlit as st

from ui_api import ApiClient
from ui_config import BASE_PATH


def render(api: ApiClient):
    st.title("Run Explorer")
    st.caption("Inspect stored runs, core trust checks, and any advisory or review signals.")
    try:
        runs = api.get(f"{BASE_PATH}/runs")
        st.dataframe(runs)
        run_id = st.text_input("Run ID")
        if run_id:
            st.json(api.get(f"{BASE_PATH}/runs/{run_id}"))
    except Exception as exc:
        st.error(str(exc))

