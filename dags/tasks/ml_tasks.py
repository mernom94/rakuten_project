#!/usr/bin/env python3
"""
ML wrapper tasks for Airflow DAG
Simple tasks that call the standalone ML scripts
"""

import subprocess
import os
import json
from datetime import datetime

def run_preprocessing_script(**context):
    """
    Airflow task: Run the standalone preprocessing script
    Calls scripts/preprocessing.py
    """
    print("Starting preprocessing script...")
    
    # Path to the preprocessing script (in Airflow container)
    script_path = "/opt/airflow/scripts/preprocessing.py"
    
    # Check if script exists
    if not os.path.exists(script_path):
        raise FileNotFoundError(f"Preprocessing script not found at {script_path}")
    
    # Run the preprocessing script
    try:
        result = subprocess.run(
            ["python", script_path],
            capture_output=True,
            text=True,
            cwd="/opt/airflow"
        )
        
        # Print script output
        print("PREPROCESSING SCRIPT OUTPUT:")
        print("=" * 50)
        print(result.stdout)
        
        if result.stderr:
            print("PREPROCESSING SCRIPT ERRORS:")
            print("=" * 50)
            print(result.stderr)
        
        # Check if script succeeded
        if result.returncode != 0:
            raise Exception(f"Preprocessing script failed with return code {result.returncode}")
        
        print("Preprocessing script completed successfully!")
        
        # Try to read metadata for next task
        try:
            metadata_path = "/opt/airflow/processed_data/latest_preprocessing.json"
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                print(f"Preprocessed {metadata.get('n_samples', 'unknown')} samples")
                return metadata
        except Exception as e:
            print(f"Could not read preprocessing metadata: {e}")
            
        return {"status": "completed", "timestamp": datetime.now().isoformat()}
        
    except Exception as e:
        print(f"Error running preprocessing script: {e}")
        raise

def run_training_script(**context):
    """
    Airflow task: Run the standalone training script
    Calls scripts/training.py
    """
    print("Starting training script...")
    
    # Path to the training script (in Airflow container)
    script_path = "/opt/airflow/scripts/training.py"
    
    # Check if script exists
    if not os.path.exists(script_path):
        raise FileNotFoundError(f"Training script not found at {script_path}")
    
    # Check if preprocessing data exists
    preprocessing_metadata_path = "/opt/airflow/processed_data/latest_preprocessing.json"
    if not os.path.exists(preprocessing_metadata_path):
        raise FileNotFoundError("No preprocessing data found. Run preprocessing task first.")
    
    # Run the training script
    try:
        result = subprocess.run(
            ["python", script_path],
            capture_output=True,
            text=True,
            cwd="/opt/airflow"
        )
        
        # Print script output
        print("TRAINING SCRIPT OUTPUT:")
        print("=" * 50)
        print(result.stdout)
        
        if result.stderr:
            print("TRAINING SCRIPT ERRORS:")
            print("=" * 50)
            print(result.stderr)
        
        # Check if script succeeded
        if result.returncode != 0:
            raise Exception(f"Training script failed with return code {result.returncode}")
        
        print("Training script completed successfully!")
        
        # Try to read model metadata
        try:
            model_metadata_path = "/opt/airflow/models/latest_gridsearch_model.json"
            if os.path.exists(model_metadata_path):
                with open(model_metadata_path, 'r') as f:
                    metadata = json.load(f)
                print(f"Best model F1 score: {metadata.get('performance', {}).get('test_f1_score', 'unknown')}")
                
                # TODO: Add MLflow logging here when marie's setup is ready
                # mlflow.log_model(metadata['model_path'])
                # mlflow.log_metrics(metadata['performance'])
                
                return metadata
        except Exception as e:
            print(f"Could not read model metadata: {e}")
            
        return {"status": "completed", "timestamp": datetime.now().isoformat()}
        
    except Exception as e:
        print(f"Error running training script: {e}")
        raise

def check_ml_environment(**context):
    """
    Optional task: Check if ML environment is ready
    Verifies required directories and dependencies exist
    """
    print("Checking ML environment...")
    
    # Check required directories
    required_dirs = [
        "/opt/airflow/scripts",
        "/opt/airflow/processed_data", 
        "/opt/airflow/models"
    ]
    
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"✓ Directory exists: {dir_path}")
        else:
            print(f"✗ Directory missing: {dir_path}")
            os.makedirs(dir_path, exist_ok=True)
            print(f"✓ Created directory: {dir_path}")
    
    # Check if scripts exist
    scripts_to_check = [
        "/opt/airflow/scripts/preprocessing.py",
        "/opt/airflow/scripts/training.py"
    ]
    
    for script_path in scripts_to_check:
        if os.path.exists(script_path):
            print(f"✓ Script exists: {script_path}")
        else:
            print(f"✗ Script missing: {script_path}")
    
    print("Environment check completed!")
    return {"status": "environment_ready"}