# streamlit_app.py
import streamlit as st

# Define pages with clear navigation structure

# INTRODUCTION PAGES
home_page = st.Page("pages/1_Homepage.py", title="1. Homepage")
team_presentation = st.Page("pages/2_Team_Presentation.py", title="2. Team Presentation")

# PROJECT PAGES
flowcharts = st.Page("pages/3_Flowcharts.py", title="3. Flowcharts (WIP)")
key_focus_areas = st.Page("pages/4_Key_Focus_Areas.py", title="4. Key Focus Areas")
fastapi_demo = st.Page("pages/5_FastAPI_Demo.py", title="5. FastAPI Demo")
monitoring_and_operations = st.Page("pages/6_Monitoring_and_Operations.py", title="6. Monitoring and Operations")
business_metrics_and_insights = st.Page("pages/7_Business_Metrics_and_Insights.py", title="7. Business Metrics and Insights")
future_improvements = st.Page("pages/8_Future_Improvements.py", title="8. Future Improvements")

# DEV PAGES
tmp = st.Page("pages/10_tmp.py", title="10. tmp")

# Create navigation with grouped pages
pg = st.navigation({
    "INTRODUCTION": [home_page, team_presentation],
    "PROJECT": [flowcharts, key_focus_areas, fastapi_demo, 
                  monitoring_and_operations, business_metrics_and_insights, future_improvements],
    "DEV": [tmp]
})

# Run the selected page
pg.run()