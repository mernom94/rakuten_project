from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
from tasks.ml_tasks import run_preprocessing_script, run_training_script, check_ml_environment

# DAG for ML pipeline - preprocessing and training
with DAG(
    dag_id='ml_pipeline',
    description='Machine Learning pipeline for Rakuten product classification',
    tags=['Rakuten', 'ML', 'MLOps'],
    schedule=None,  # Manual trigger for now
    default_args={
        'owner': 'airflow',
        'start_date': datetime(2025, 6, 19),
        'retries': 1,
    },
    catchup=False,
    doc_md="""
    # ML Pipeline DAG
    
    This DAG orchestrates the machine learning pipeline for Rakuten product classification:
    
    ## Tasks:
    1. **check_environment**: Verifies required directories and scripts exist
    2. **run_preprocessing**: Calls scripts/preprocessing.py to process text data
    3. **run_training**: Calls scripts/training.py to train ML models
    
    ## Data Flow:
    - Input: Raw text data from PostgreSQL (via your preprocessing script)
    - Processing: Your standalone preprocessing script
    - Training: Your standalone training script with GridSearchCV
    - Output: Models saved by your scripts, ready for MLflow integration
    
    ## MLflow Integration:
    - Ready for MLflow logging (when marie's setup is complete)
    - Models and metrics will be tracked in MLflow
    
    ## Usage:
    - Trigger manually for now
    - Can be scheduled or triggered by other DAGs later
    """
) as dag:
    
    # Task 0: Check environment (optional)
    environment_check = PythonOperator(
        task_id='check_environment',
        python_callable=check_ml_environment,
        doc_md="""
        ## Environment Check Task
        
        Verifies that the ML environment is properly set up:
        - Checks required directories exist
        - Verifies ML scripts are present
        - Creates missing directories if needed
        """
    )
    
    # Task 1: Run preprocessing script
    preprocess_task = PythonOperator(
        task_id='run_preprocessing',
        python_callable=run_preprocessing_script,
        doc_md="""
        ## Preprocessing Task
        
        Runs your standalone preprocessing script (scripts/preprocessing.py):
        - Loads raw text data from PostgreSQL
        - Performs text preprocessing and feature extraction
        - Saves processed data for training
        
        **Your script handles:**
        - Text cleaning and French stopwords
        - TF-IDF feature extraction  
        - Multiple text versions (raw, classical, bert-ready)
        """
    )
    
    # Task 2: Run training script
    train_task = PythonOperator(
        task_id='run_training',
        python_callable=run_training_script,
        doc_md="""
        ## Training Task
        
        Runs your standalone training script (scripts/training.py):
        - Uses preprocessed data from previous task
        - Performs GridSearchCV with multiple ML algorithms
        - Saves best model and evaluation metrics
        
        **Your script handles:**
        - Random Forest, Logistic Regression, SVM, XGBoost
        - Hyperparameter tuning with GridSearchCV
        - Model evaluation and saving
        
        **Future:** Ready for MLflow integration when marie's setup is complete
        """
    )
    
    # Define task dependencies
    environment_check >> preprocess_task >> train_task