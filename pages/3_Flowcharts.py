# 3_Flowcharts.py
import streamlit as st
from streamlit_mermaid import st_mermaid
from containers.rakuten_st.streamlit_utils import add_pagination_and_footer

st.set_page_config(
    page_title="MAY25 BDS // Flowcharts",
    page_icon="images/logos/rakuten-favicon.ico",
    layout="wide",
)

st.progress(3 / 8)
st.title("Flowcharts")

st.image("images/screenshots/flowchart.png", caption="Flowcharts for the MAY25 BDS project [PLACEHOLDER]", use_container_width=True)

# Pagination and footer
st.markdown("---")
add_pagination_and_footer("pages/3_Flowcharts.py")