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

# Create tabs for the three main sections
tab1, tab2, tab3 = st.tabs(["Introduction", "Effect of MLOps Architecture", "Critical Learnings"])

with tab1:
    st.markdown("""
    ### Current Implementation
    Our MLOps system currently uses **technical metrics** (F1 score, accuracy) to determine model quality and deployment decisions. The system automatically compares new models against production models and only promotes improvements.

    ### Business Logic Evolution
    While technical metrics ensure model quality, **future implementations could integrate business logic** for more sophisticated decision-making:

    **Example: Customer Churn Prevention**
    - Current: Deploy if F1 score > 85%
    - Future: Deploy if F1 score > 85% AND predicted churn reduction > 2% AND customer lifetime value impact > $50k

    **Example: Revenue Optimization**
    - Current: Accuracy-based deployment
    - Future: Deploy models that maximize revenue per prediction, even with slightly lower accuracy

    **Example: Operational Constraints** 
    - Current: Technical performance only
    - Future: Consider deployment costs, infrastructure capacity, and business seasonality

    This evolution from **technical-first to business-first decision making** represents the maturity path for MLOps systems.
    """)

with tab2:
    st.markdown("""
    ### Operational Efficiency Gains
    - **Automated Retraining**: Only retrains when eval_f1 > current_production_f1 → reduces computational costs
    - **Quality Gates**: Prevents deployment of worse models → reduces operational risk  
    - **Docker Containerization**: Consistent environments → reduces deployment failures
    - **MLflow Integration**: Automated model comparison → reduces manual decision-making
    - **Pipeline Orchestration**: Airflow handles complex dependencies → reduces coordination overhead

    ### Risk Mitigation
    - **Production Stability**: Quality gates prevent model degradation automatically
    - **Model Versioning**: MLflow registry enables instant rollback capability
    - **Real-time Monitoring**: Grafana dashboards provide system visibility → enables proactive issue detection
    - **Automated Pipelines**: Airflow orchestration → reduces human error in workflow execution
    - **Containerized Deployment**: Docker isolation → reduces environment-related failures

    ### Scalability Benefits
    - **Containerized ML**: Docker enables consistent horizontal scaling
    - **Modular Architecture**: FastAPI, MLflow, PostgreSQL can scale independently
    - **Automated Processing**: Handles increasing data volumes without manual intervention
    - **Pipeline Reusability**: Airflow DAGs can be extended for additional product categories
    """)

with tab3:
    st.markdown("""
    ### Technical Architecture Insights
    - **Container orchestration complexity**: Docker Compose alone won't build custom containers - setup scripts are needed for complete ML pipeline deployment
    - **Resource management**: Disk space and memory constraints significantly affect complex ML container builds in production environments
    - **Service dependencies**: Monitoring services (Grafana/Prometheus) can fail silently during resource-intensive operations

    ### MLOps Implementation Lessons
    - **Quality gates vs. training failures**: Critical distinction between "training failed" and "model not promoted due to quality criteria" - an important production consideration
    - **MLflow model registry**: Understanding the difference between simple metrics-based selection vs. proper model registry stages (choosing the more professional approach)

    ### Team Collaboration & Version Control
    - **Branch management**: Managing team changes while working on different features requires careful coordination
    - **README accuracy**: Documentation needs constant updates to reflect actual implementation, especially startup procedures
    - **Team communication**: Importance of checking with teammates about infrastructure changes and pending PRs

    ### Production-Ready Thinking
    - **Business vs. technical metrics**: Evolution from pure technical metrics toward future business logic integration (such as churn prevention scenarios)
    """)

# Pagination and footer
st.markdown("---")
add_pagination_and_footer("pages/7_Business_Metrics_and_Insights.py")
