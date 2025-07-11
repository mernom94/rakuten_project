# pages/7_Business_Metrics_and_Insights 
import streamlit as st
from containers.rakuten_st.streamlit_utils import add_pagination_and_footer

st.set_page_config(
    page_title="MAY25 BDS // Business Metrics and Insights",
    page_icon="images/logos/rakuten-favicon.ico",
    layout="wide",
)

st.progress(7 / 8)
st.title("Business Metrics and Insights ")

# Pagination and footer
st.markdown("---")
add_pagination_and_footer("pages/7_Business_Metrics_and_Insights.py")