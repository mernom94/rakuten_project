# pages/02_Team_Presentation.py
import streamlit as st
from containers.rakuten_st.streamlit_utils import add_pagination_and_footer


# Page configuration
st.set_page_config(
    page_title="MAY25 BDS // Team Presentation",
    page_icon="images/logos/rakuten-favicon.ico",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Logo display
st.logo(image="images/logos/rakuten-logo-red-wide.svg", size="large", icon_image="images/logos/rakuten-logo-red-square.svg")

st.progress(2 / 8)
st.title("Team Presentation")

# Create three columns for team members
tp_col1, tp_col2, tp_col3, tp_col4 = st.columns(4)  # tp_ as in "team presentation"


with tp_col1:
    st.image("images/profile_pictures/marie_erno-moller.png", use_container_width=True)
    st.info("Marie Ernø-Møller")

    # Display primary contribution information directly
    st.write("**Primary Contributions:**")
    st.write("• MLFlow tracking")
    st.write("• Evidently drift detection")
    st.write("• Prometheus scraping & Grafana visualization")

    with st.expander("… more info"):
        st.markdown(
            """
            **Marie Ernø-Møller**
            ---
            
            **Former positions:**
            - Position #1
            - Position #2
            - Position #3
            
            **Skills & Expertise:**
            - Skill #1
            - Skill #2
            - Skill #3

            **Links:**
            - [GitHub Profile](https://github.com/mernom94)
            - [LinkedIn Profile](https://www.linkedin.com/in/marie-ern%C3%B8-m%C3%B8ller-80081b253/)
            """
        )


with tp_col2:
    st.image("images/profile_pictures/peter_stieg.jpg", use_container_width=True)
    st.info("Peter Stieg")

    # Display primary contribution information directly
    st.write("**Primary Contributions:**")
    st.write("• Preprocessing of text data")
    st.write("• Streamlit presentation")
    st.write("• Refactoring")

    with st.expander("… more info"):
        st.markdown(
            """
            **Peter Stieg**
            ---
            

            **Former positions:**
            - Marketing Director
            - Head of Marketing
            - COO
            
            **Skills & Expertise:**
            - Project Management 
            - User Experience
            - Marketing
            
            **Links:**
            - [GitHub Profile](https://github.com/peterstieg/)
            - [LinkedIn Profile](https://www.linkedin.com/in/PeterStieg/)
            """
        )


with tp_col3:
    st.image("images/profile_pictures/qi_bao.png", use_container_width=True)
    st.info("Qi Bao")

    # Display primary contribution information directly
    st.write("**Primary Contributions:**")
    st.write("• Foundational Architecture")
    st.write("• Airflow Orchestration")
    st.write("• Project Management")

    with st.expander("… more info"):
        st.markdown(
        """
        **Qi Bao**
        ---
        

        **Former positions:**
        - Go-To-Market Manager in telecommunications
        
        **Skills & Expertise:**
        - Virtual Machines
        - Web scraping

        
        **Links:**
        - [GitHub Profile](https://github.com/Pockyee)
        - [LinkedIn Profile ???](https://www.linkedin.com/)
        """
        )


with tp_col4:
    st.image("images/profile_pictures/robert_wilson.jpg", use_container_width=True)
    st.info("Robert Wilson")

    # Display primary contribution information directly
    st.write("**Primary Contributions:**")
    st.write("• Model Development and Evaluation")
    st.write("• Preprocessing of text")
    st.write("• FastAPI integration")

    with st.expander("… more info"):
        st.markdown(
            """
            **Robert Wilson**
            ---
            
            **Former positions:**
            - Director of Sales and Marketing
            - Senior Account Executive
            - Sales Manager
            
            **Skills & Expertise:**
            - SaaS B2B Enterprise Sales
            - Performance Marketing
            - Pre-Sales
            
            **Links:**
            - [GitHub Profile](https://github.com/Wilsbert12)
            - [LinkedIn Profile](https://www.linkedin.com/in/robert-wilson-17081983/)
            """
        )


# Pagination and footer
st.markdown("---")
add_pagination_and_footer("pages/2_Team_Presentation.py")