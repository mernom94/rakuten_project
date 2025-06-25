from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.operators.python import PythonOperator
from datetime import datetime
import os

# DAG for ML pipeline using DockerOperator
with DAG(
    dag_id='ml_pipeline_docker',
    description='Machine Learning pipeline using Docker containers for Rakuten product classification',
    tags=['Rakuten', 'ML', 'MLOps', 'Docker'],
    schedule=None,  # Manual trigger for now
    default_args={
        'owner': 'airflow',
        'start_date': datetime(2025, 6, 19),
        'retries': 1,
    },
    catchup=False,
    doc_md="""
    # ML Pipeline DAG (Docker Version)
    
    This DAG orchestrates the machine learning pipeline using Docker containers:
    
    ## Tasks:
    1. **check_environment**: Verifies required directories exist
    2. **run_preprocessing_docker**: Runs preprocessing.py in ML Docker container
    3. **run_training_docker**: Runs training.py in ML Docker container
    
    ## Benefits of Docker Approach:
    - Isolated ML environment with all dependencies
    - Consistent execution across different environments  
    - Easy to version and scale ML components
    - No dependency conflicts with Airflow
    
    ## Data Flow:
    - Input: Raw text data from PostgreSQL
    - Processing: ML container runs preprocessing script
    - Training: ML container runs training script with GridSearchCV
    - Output: Models saved to shared volume, ready for MLflow integration
    
    ## Volumes:
    - Project directory mounted to /app in container
    - Processed data and models persist between runs
    """
) as dag:
    
    def check_directories():
        """Ensure required directories exist on host"""
        import os
        
        # Directories that need to exist for volume mounting
        required_dirs = [
            '/opt/airflow/processed_data',
            '/opt/airflow/models',
            '/opt/airflow/scripts'
        ]
        
        for dir_path in required_dirs:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path, exist_ok=True)
                print(f"Created directory: {dir_path}")
            else:
                print(f"Directory exists: {dir_path}")
        
        return "Directories ready"
    
    # Task 0: Check environment
    environment_check = PythonOperator(
        task_id='check_environment',
        python_callable=check_directories,
        doc_md="""
        ## Environment Check Task
        
        Ensures required directories exist on the host for volume mounting:
        - /opt/airflow/processed_data (for preprocessed features)
        - /opt/airflow/models (for trained models)  
        - /opt/airflow/scripts (for ML scripts)
        """
    )
    
    # Task 1: Run preprocessing in Docker container
    preprocessing_docker = DockerOperator(
        task_id='run_preprocessing_docker',
        image='rakuten-ml:latest',  # Your ML container image
        command='python scripts/preprocessing.py',
        docker_url='unix://var/run/docker.sock',
        network_mode='rakuten_project_default',  # Connect to your project's Docker network
        volumes=[
            '/opt/airflow/processed_data:/app/processed_data',
            '/opt/airflow/models:/app/models',
            '/opt/airflow/scripts:/app/scripts'
        ],
        environment={
            'PYTHONPATH': '/app',
            'PYTHONUNBUFFERED': '1'
        },
        auto_remove=True,
        doc_md="""
        ## Preprocessing Docker Task
        
        Runs preprocessing.py in an isolated ML container:
        - Container has all ML dependencies (sklearn, nltk, etc.)
        - Connects to PostgreSQL via Docker network
        - Mounts volumes for data persistence
        - Processes French text and creates TF-IDF features
        
        **Container Setup:**
        - Image: rakuten-ml:latest
        - Network: rakuten_project_default (access to postgres)
        - Volumes: Maps host directories to container
        """
    )
    
    # Task 2: Run training in Docker container  
    training_docker = DockerOperator(
        task_id='run_training_docker',
        image='rakuten-ml:latest',  # Same ML container image
        command='python scripts/training.py',
        docker_url='unix://var/run/docker.sock',
        network_mode='rakuten_project_default',  # Connect to your project's Docker network
        volumes=[
            '/opt/airflow/processed_data:/app/processed_data',
            '/opt/airflow/models:/app/models',
            '/opt/airflow/scripts:/app/scripts'
        ],
        environment={
            'PYTHONPATH': '/app',
            'PYTHONUNBUFFERED': '1'
        },
        auto_remove=True,
        doc_md="""
        ## Training Docker Task
        
        Runs training.py in the ML container:
        - Uses preprocessed data from previous task
        - Performs GridSearchCV with multiple algorithms
        - Saves best models and metadata
        - Ready for MLflow integration
        
        **Algorithms Tested:**
        - Random Forest, Logistic Regression, SVM, XGBoost
        - Hyperparameter tuning with GridSearchCV
        - Evaluation using weighted F1 score
        """
    )
    
    # Define task dependencies
    environment_check >> preprocessing_docker >> training_docker
