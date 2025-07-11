# pages/4_Key_Focus_Areas.py
import streamlit as st
from containers.rakuten_st.streamlit_utils import add_pagination_and_footer

import pandas as pd


# Page configuration
st.set_page_config(
    page_title="MAY25 BMLOPS // Key Focus Areas",
    page_icon="images/logos/rakuten-favicon.ico",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Logo display
st.logo(image="images/logos/rakuten-logo-red-wide.svg", size="large", icon_image="images/logos/rakuten-logo-red-square.svg")


# Container infrastructure data
container_data = {
    'CONTAINER': [
        'Docker',
        'PostgreSQL', 
        'Redis',
        'MinIO',
        'Apache Airflow',
        'MLflow',
        'FastAPI',
        'pgAdmin'
    ],
    'PORT': [
        '-',
        '5432',
        '6379', 
        '9000 (API), 9001 (Console)',
        '8080',
        '5001',
        '8000',
        '8081'
    ],
    'USAGE': [
        'Containerization platform for all services',
        'Database for Airflow metadata and Rakuten product data storage',
        'Message broker for Airflow Celery executor and caching',
        'Object storage for product images and ML artifacts',
        'Workflow orchestration and ML pipeline scheduling',
        'ML experiment tracking, model registry and versioning',
        'REST API for model serving and predictions',
        'Web-based PostgreSQL database administration interface'
    ]
}

# Volume data
volume_data = {
    'VOLUME': [
        'postgres-db-volume',
        'minio_data', 
        'mlflow_data',
        'pgadmin_data',
        'dags (bind mount)',
        'logs (bind mount)',
        'config (bind mount)',
        'raw_data (bind mount)',
        'docker.sock (bind mount)'
    ],
    'TYPE': [
        'Named Volume',
        'Named Volume',
        'Named Volume', 
        'Named Volume',
        'Host Bind Mount',
        'Host Bind Mount',
        'Host Bind Mount',
        'Host Bind Mount',
        'Host Bind Mount'
    ],
    'PURPOSE': [
        'Database persistence',
        'Object storage for images',
        'ML artifacts storage',
        'Admin interface data',
        'Airflow DAG files',
        'Task execution logs',
        'Airflow configuration',
        'Raw dataset storage',
        'Docker daemon access'
    ]
}

# DAG data
dag_data = {
    'DAG': [
        'prepare_data',
        'ml_pipeline_docker',
        'reset_data',
        'test_dag'
    ],
    'SCHEDULE': [
        'Every 5 minutes',
        'Every 5 minutes',
        'Manual trigger only',
        'Manual trigger only'
    ],
    'PURPOSE': [
        'Load Rakuten data to PostgreSQL incrementally',
        'Execute containerized ML training and preprocessing',
        'Reset and clean test dataset tables',
        'Test database connectivity and system metrics'
    ]
}

# Custom image data
custom_images_data = {
    'IMAGE': [
        'rakuten-ml',
        'rakuten-fastapi',
        'rakuten_st'
    ],
    'BASE': [
        'python:3.9-slim',
        'python:3.11-slim',
        'python:3.11-slim'
    ],
    'PURPOSE': [
        'ML preprocessing and training',
        'API serving trained model',
        'Project presentation'
    ]
}

# Create DataFrames
container_df = pd.DataFrame(container_data)
volume_df = pd.DataFrame(volume_data)
dag_df = pd.DataFrame(dag_data)
custom_images_df = pd.DataFrame(custom_images_data)

st.progress(4 / 8)
st.title("Key Focus Areas")

# Project Outline content
container_tab, volume_tab, dag_tab, custom_image_tab = st.tabs(["Containers", "Volumes", "DAGs", "Custom images"]
)

with container_tab:
    st.markdown("""
    This table shows the container infrastructure for a complete MLOps pipeline setup, 
    organized in logical order from foundation to service layers.
    """)

    # Display the table
    st.dataframe(
        container_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "CONTAINER": st.column_config.TextColumn(
                "CONTAINER",
                width="small",
            ),
            "PORT": st.column_config.TextColumn(
                "PORT",
                width="small",
            ),
            "USAGE": st.column_config.TextColumn(
                "USAGE",
                width="large",
            ),
        }
    )

with volume_tab:
    st.markdown("""
    This table shows the storage infrastructure for the complete MLOps pipeline, organized by volume type from persistent data storage to development and operational bind mounts.
    """)

    # Display the table
    st.dataframe(
        volume_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "VOLUME": st.column_config.TextColumn(
                "VOLUME",
                width="small",
            ),
            "TYPE": st.column_config.TextColumn(
                "TYPE",
                width="small",
            ),
            "PURPOSE": st.column_config.TextColumn(
                "PURPOSE",
                width="large",
            ),
        }
    )

with dag_tab:
    st.markdown("""
    This table shows the workflow infrastructure for the complete MLOps pipeline, organized by data flow from acquisition to model training.
                """)

    # Display the table
    st.dataframe(
        dag_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "DAG": st.column_config.TextColumn(
                "DAG",
                width="small",
            ),
            "SCHEDULE": st.column_config.TextColumn(
                "SCHEDULE",
                width="small",
            ),
            "PURPOSE": st.column_config.TextColumn(
                "PURPOSE",
                width="large",
            ),
        }
    )

with custom_image_tab:
    st.markdown("""
                This table shows the custom images used in the MLOps pipeline, organized by purpose from ML training to API serving.
                """)
    
    # Display the table
    st.dataframe(
        custom_images_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "IMAGE": st.column_config.TextColumn(
                "IMAGE",
                width="small",
            ),
            "BASE": st.column_config.TextColumn(
                "BASE",
                width="small",
            ),
            "PURPOSE": st.column_config.TextColumn(
                "PURPOSE",
                width="large",
            ),
        }
    )

    
# Pagination and footer
st.markdown("---")
add_pagination_and_footer("pages/4_Key_Focus_Areas.py")