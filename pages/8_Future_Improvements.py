# pages/8_Future_Improvements
import streamlit as st
from containers.rakuten_st.streamlit_utils import add_pagination_and_footer

st.set_page_config(
    page_title="MAY25 BDS // Future Improvements",
    page_icon="images/logos/rakuten-favicon.ico",
    layout="wide",
)

st.progress(8 / 8)
st.title("Future Improvements")

# Pagination and footer
st.markdown("---")
add_pagination_and_footer("pages/8_Future_Improvements.py")