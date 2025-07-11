# pages/01_Home.py
import streamlit as st
from containers.rakuten_st.streamlit_utils import add_pagination_and_footer


# Page configuration
st.set_page_config(
    page_title="MAY25 BMLOPS // Rakuten: Classification of eCommmerce products",
    page_icon="images/logos/rakuten-favicon.ico",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Logo display
st.logo(image="images/logos/rakuten-logo-red-wide.svg", size="large", icon_image="images/logos/rakuten-logo-red-square.svg")

st.progress(1 / 8)
st.title("MAY25 BMLOPS // Rakuten")

# Home page content
st.write("## eCommerce Products Classification Project")
st.markdown(
    """
    This Streamlit app is part of the final project for **_DataScientist_**'s training in **Machine Learning Operations** of the cohort **MAY25 BMLOPS**.

    **Primary Objective:** Build a production-ready machine learning infrastructure that can reliably process French product titles and descriptions and classify them into appropriate categories while maintaining operational excellence during the entire machine learning lifecycle from data ingestion and preprocessing to model deployment and monitoring.
    
    _Please note:_ This project does not focus on the model's accuracy and performance but rather on the MLOps **pipeline's robustness and scalability**.
    
    Use the sidebar or pagination to browse through the presentation of the project and the team, ...
    
    **:material/folder_code: GitHub Repository:** [may25_bds_classification-of-rakuten-e-commerce-products](https://github.com/PeterStieg/may25_bds_classification-of-rakuten-e-commerce-products/)
    
    """
)

# Pagination and footer
st.markdown("---")
add_pagination_and_footer("pages/1_Home.py")
