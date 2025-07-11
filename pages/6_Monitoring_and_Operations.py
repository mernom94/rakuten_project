# pages/6_Monitoring_and_Operations
import streamlit as st
from containers.rakuten_st.streamlit_utils import add_pagination_and_footer

st.set_page_config(
    page_title="MAY25 BDS // Monitoring and Operations",
    page_icon="images/logos/rakuten-favicon.ico",
    layout="wide",
)

st.progress(6 / 8)
st.title("Monitoring and Operations")

# Pagination and footer
st.markdown("---")
add_pagination_and_footer("pages/6_Monitoring_and_Operations.py")