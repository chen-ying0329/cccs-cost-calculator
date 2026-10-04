from __future__ import annotations

import streamlit as st


chinese_page = st.Page(
    "pages/Chinese.py",
    title="中文版",
    icon="🇨🇳",
    url_path="Chinese",
    visibility="hidden",
)
english_page = st.Page(
    "pages/English_Version.py",
    title="English",
    icon="🌐",
    url_path="English",
    visibility="hidden",
)


def open_default_page() -> None:
    st.switch_page(chinese_page)


default_page = st.Page(
    open_default_page,
    title="CCCS–Cost",
    default=True,
    visibility="hidden",
)

current_page = st.navigation(
    [default_page, chinese_page, english_page],
    position="hidden",
)
current_page.run()
