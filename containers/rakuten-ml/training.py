#!/usr/bin/env python3
"""
Training script for Rakuten product classification
Loads preprocessed data and trains ML models using GridSearchCV
Based on proven approach from previous project
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, f1_score
import pickle
import json
import os
from datetime import datetime
from scipy.sparse import load_npz
from collections import Counter
import mlflow
import mlflow.sklearn
from mlflow.exceptions import RestException

# Directories
PROCESSED_DATA_DIR = 'processed_data'
MODELS_DIR = 'models'

try:
    from statsd import StatsClient
    statsd = StatsClient(host="statsd-exporter", port=8125, prefix="mlflow")
except ImportError:
    statsd = None
    print("⚠️  'statsd' package not installed. Metrics will not be exported to Prometheus.")

mlflow.set_tracking_uri("http://mlflow:5000")
mlflow.set_experiment("RakutenTraining")

def load_latest_processed_data():
    """Load the most recent preprocessed data"""
    print("Loading preprocessed data...")
    
    # Load metadata to get file paths
    metadata_path = os.path.join(PROCESSED_DATA_DIR, 'latest_preprocessing.json')
    
    if not os.path.exists(metadata_path):
        raise FileNotFoundError("No preprocessed data found. Please run preprocessing.py first.")
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Load the processed text DataFrame (contains all text versions)
    text_df = pd.read_csv(metadata['text_path'])
    
    print(f"Loaded {len(text_df)} samples")
    print(f"Available text versions: {list(text_df.columns)}")
    
    return text_df, metadata

def load_eval_data():
    with open(os.path.join(PROCESSED_DATA_DIR, 'preprocessing_metadata_test.json'), 'r') as f:
        metadata = json.load(f)
    text_df = pd.read_csv(metadata['text_path'])
    print(f"Loaded {len(text_df)} samples")
    print(f"Available text versions: {list(text_df.columns)}")
    return text_df, metadata

def print_class_distribution(y, dataset_name):
    """Print class distribution for analysis"""
    # Count the occurrences of each class
    class_counts = Counter(y)
    total_samples = sum(class_counts.values())
    
    # Calculate the relative distribution in percentages
    relative_distribution = {cls: (count / total_samples) * 100 for cls, count in class_counts.items()}
    
    # Print the distribution
    print(f"Class distribution in {dataset_name} (in %):")
    for cls, percentage in sorted(relative_distribution.items()):
        print(f"  Class {cls}: {percentage:.2f}%")
    print()

def create_stratified_split(X, y, test_size=0.2, random_state=42):
    """Create stratified train/test split"""
    print("Creating stratified train/test split...")
    
    # Initialize stratified shuffle split
    stratified_split = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    
    # Perform the split
    for train_index, test_index in stratified_split.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    # Print class distributions
    print_class_distribution(y, "Original Dataset")
    print_class_distribution(y_train, "Training Set")
    print_class_distribution(y_test, "Test Set")
    
    return X_train, X_test, y_train, y_test

def create_pipeline_and_param_grid():
    """Create pipeline and parameter grid based on proven approach"""
    print("Setting up pipeline and parameter grid...")
    
    # Define the pipeline (placeholder components)
    pipeline = Pipeline([
        ('vectorizer', CountVectorizer()),  # Placeholder for vectorizer
        ('classifier', RandomForestClassifier(random_state=42))  # Placeholder for classifier
    ])
    
    # Define the parameter grid based on proven approach
    param_grid = [
        # Random Forest parameters
        {
            'vectorizer': [CountVectorizer(), TfidfVectorizer()],
            'vectorizer__max_features': [5000],
            'vectorizer__ngram_range': [(1, 1)],
            'classifier': [RandomForestClassifier(random_state=42)],
            'classifier__n_estimators': [50, 100],
            'classifier__max_depth': [10, 20],
        },
        # Logistic Regression parameters
        {
            'vectorizer': [CountVectorizer(), TfidfVectorizer()],
            'vectorizer__max_features': [5000],
            'vectorizer__ngram_range': [(1, 1)],
            'classifier': [LogisticRegression(random_state=42, max_iter=1000)],
            'classifier__C': [0.1, 1, 10],
            'classifier__penalty': ['l2'],
        },
        # SVM parameters
        {
            'vectorizer': [CountVectorizer(), TfidfVectorizer()],
            'vectorizer__max_features': [5000],
            'vectorizer__ngram_range': [(1, 1)],
            'classifier': [SVC(random_state=42)],
            'classifier__C': [0.1, 1, 10],
            'classifier__kernel': ['linear'],  # Only linear kernel for speed
        },
        # XGBoost parameters
        {
            'vectorizer': [CountVectorizer(), TfidfVectorizer()],
            'vectorizer__max_features': [5000],
            'vectorizer__ngram_range': [(1, 1)],
            'classifier': [XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')],
            'classifier__n_estimators': [50],
            'classifier__max_depth': [3],
            'classifier__learning_rate': [0.1],
        }
    ]
    
    return pipeline, param_grid

def train_with_gridsearch(X_train, X_test, y_train, y_test, pipeline, param_grid, X_eval, y_eval):
    """Train models using GridSearchCV"""
    print("Starting GridSearchCV training...")
    print("This may take some time depending on the parameter grid size...")
    
    # Encode the labels
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)
    y_eval_encoded = label_encoder.transform(y_eval)  # added for eval
    
    # Perform grid search with weighted F1 score
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        scoring='f1_weighted',  # Use weighted F1 score for evaluation
        cv=3,  # Use 3-fold cross-validation for faster computation
        n_jobs=-1,  # Use all available CPU cores
        verbose=1  # Show progress
    )
    
    # Fit the grid search
    print("Fitting GridSearchCV...")
    grid_search.fit(X_train, y_train_encoded)
    
    # Print results
    print("=" * 60)
    print("GRIDSEARCH RESULTS:")
    print("=" * 60)
    print("Best Parameters:", grid_search.best_params_)
    print("Best Cross-Validation Score:", grid_search.best_score_)

            # Log hyperparameters
    mlflow.log_params(grid_search.best_params_)
    
    
    # Log the trained model
    mlflow.sklearn.log_model(grid_search.best_estimator_, "model")
    
    # Predict on the test set
    y_test_pred_encoded = grid_search.best_estimator_.predict(X_test)
    y_test_pred = label_encoder.inverse_transform(y_test_pred_encoded)
    
    # Calculate test metrics
    test_f1 = f1_score(y_test, y_test_pred, average='weighted')
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    print(f"Test Set Weighted F1 Score: {test_f1:.4f}")
    print(f"Test Set Accuracy: {test_accuracy:.4f}")
    print("\nClassification Report on Test Set:")
    print(classification_report(y_test, y_test_pred))
    
    # Predict on the evaluation set
    y_eval_pred_encoded = grid_search.best_estimator_.predict(X_eval)
    y_eval_pred = label_encoder.inverse_transform(y_eval_pred_encoded)

    # Calculate eval metrics
    eval_f1 = f1_score(y_eval, y_eval_pred, average='weighted')
    eval_accuracy = accuracy_score(y_eval, y_eval_pred)
    
    n_samples = len(X_train) + len(X_test)
    
    print(f"\nEval Set Weighted F1 Score: {eval_f1:.4f}")
    print(f"Eval Set Accuracy: {eval_accuracy:.4f}")
    print("\nClassification Report on Eval Set:")
    print(classification_report(y_eval, y_eval_pred))
    
    # Prepare results
    results = {
        'best_params': grid_search.best_params_,
        'best_cv_score': grid_search.best_score_,
        'test_f1_score': test_f1,
        'test_accuracy': test_accuracy,
        'eval_f1_score': eval_f1,
        'eval_accuracy': eval_accuracy,
        'label_encoder': label_encoder,
        'best_estimator': grid_search.best_estimator_
    }
            # Log performance metrics
    mlflow.log_metrics({
        "cv_score": float(grid_search.best_score_),
        "test_f1": float(test_f1),
        "test_accuracy": float(test_accuracy),
        "eval_f1": float(eval_f1),
        "eval_accuracy": float(eval_accuracy),
        "n_samples": int(n_samples)
    })
    
    # Send metrics to StatsD (Prometheus)
    if statsd:
        statsd.incr("experiment_run_total")  # Increment total runs
        statsd.gauge("accuracy", test_accuracy)
        statsd.gauge("f1", test_f1)
        statsd.gauge("cv", grid_search.best_score_)
    
    return results

def save_gridsearch_model(results, preprocessing_metadata, text_version='text_classical'):
    """Save the best model from GridSearchCV"""
    print("Saving best model from GridSearchCV...")
    
    # Create timestamp for versioning
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save best estimator (pipeline)
    model_path = os.path.join(MODELS_DIR, f'best_model_gridsearch_{timestamp}.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(results['best_estimator'], f)
    
    # Save label encoder
    encoder_path = os.path.join(MODELS_DIR, f'label_encoder_{timestamp}.pkl')
    with open(encoder_path, 'wb') as f:
        pickle.dump(results['label_encoder'], f)
    
    # Create comprehensive metadata
    model_metadata = {
        'timestamp': timestamp,
        'model_path': model_path,
        'encoder_path': encoder_path,
        'model_type': 'gridsearch_best',
        'text_version_used': text_version,
        # 'best_params': results['best_params'],
        # 'performance': {
        #     'best_cv_score': results['best_cv_score'],
        #     'test_f1_score': results['test_f1_score'],
        #     'test_accuracy': results['test_accuracy']
        # },
        'training_approach': 'gridsearch_with_pipeline',
        'scoring_metric': 'f1_weighted',
        'cv_folds': 3,
        'preprocessing_info': {
            'preprocessing_timestamp': preprocessing_metadata['timestamp'],
            'n_features_used': preprocessing_metadata['n_features'],
            'n_samples': preprocessing_metadata['n_samples']
        }
    }
    
    # Save model metadata
    metadata_path = os.path.join(MODELS_DIR, f'gridsearch_metadata_{timestamp}.json')
    with open(metadata_path, 'w') as f:
        metadata_for_json = model_metadata.copy()
        metadata_for_json.pop('best_estimator', None)  # Remove the SVM model object

        json.dump(metadata_for_json, f, indent=2)
    
    # Save as "latest" for easy access
    latest_model_path = os.path.join(MODELS_DIR, 'latest_gridsearch_model.json')
    with open(latest_model_path, 'w') as f:
        metadata_for_json = model_metadata.copy()
        metadata_for_json.pop('best_estimator', None)  # Remove the SVM model object

        json.dump(metadata_for_json, f, indent=2)
    
    print(f"Best model saved: {model_path}")
    print(f"Label encoder saved: {encoder_path}")
    print(f"Metadata saved: {metadata_path}")
    
    return model_metadata

def register_if_best_model(results, registered_model_name="TheBestModelTillNow"):
    """
    Register model only if eval_f1 is better than current production model
    """
    client = mlflow.tracking.MlflowClient()
    new_eval_f1 = results["eval_f1_score"]
    new_eval_accuracy = results["eval_accuracy"]

    try:
        # Get latest production model
        latest_versions = client.get_latest_versions(registered_model_name, stages=["Production"])
        if latest_versions:
            current_model = latest_versions[0]
            current_run_id = current_model.run_id
            current_run = client.get_run(current_run_id)
            current_eval_f1 = float(current_run.data.metrics.get("eval_f1", 0.0))
            print(f"Current registered eval_f1: {current_eval_f1:.4f}")
        else:
            current_eval_f1 = 0.0
            print("No production model registered yet.")

    except RestException:
        print(f"No registered model named '{registered_model_name}' found. Creating new one...")
        current_eval_f1 = 0.0

    if new_eval_f1 > current_eval_f1:
        print(f"New model eval_f1 ({new_eval_f1:.4f}) is better. Registering model...")
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
        result = mlflow.register_model(model_uri, registered_model_name)

        # Record model key metrics as tag
        client.set_model_version_tag(
            name=registered_model_name,
            version=result.version,
            key="eval_f1",
            value=f"{new_eval_f1:.4f}"
        )
        client.set_model_version_tag(
            name=registered_model_name,
            version=result.version,
            key="eval_accuracy",
            value=f"{new_eval_accuracy:.4f}"
        )
        
        # Transition to production
        client.transition_model_version_stage(
            name=registered_model_name,
            version=result.version,
            stage="Production",
            archive_existing_versions=True
        )
        print(f"Model registered and transitioned to Production (version {result.version})")
        
        best_model_path = os.path.join(MODELS_DIR, 'the_best_model.pkl')
        best_encoder_path = os.path.join(MODELS_DIR, 'the_label_encoder.pkl')
        
        with open(best_model_path, 'wb') as f_model:
            pickle.dump(results['best_estimator'], f_model)

        with open(best_encoder_path, 'wb') as f_enc:
            pickle.dump(results['label_encoder'], f_enc)
    else:
        print(f"New model eval_f1 ({new_eval_f1:.4f}) is not better than current ({current_eval_f1:.4f}). Not registering.")

def main():
    """Main training pipeline with GridSearchCV"""
    print("Starting GridSearchCV training pipeline...")
    print("Based on proven approach from previous project")
    print("=" * 60)
    with mlflow.start_run():
        
        try:
            # Step 1: Load preprocessed data
            text_df, preprocessing_metadata = load_latest_processed_data()
            text_df_eval, preprocessing_metadata_eval = load_eval_data()
            
            # Step 2: Extract features and target (using classical ML text)
            X = text_df['text_classical']  # Use heavily preprocessed text
            y = text_df['prdtypecode']
            
            X_eval = text_df_eval['text_classical']
            y_eval = text_df_eval['prdtypecode']
            
            print(f"Using text_classical for training ({len(X)} samples)")
            
            # Step 3: Create stratified train/test split
            X_train, X_test, y_train, y_test = create_stratified_split(X, y)
            
            # Step 4: Set up pipeline and parameter grid
            pipeline, param_grid = create_pipeline_and_param_grid()
            
            # Step 5: Train with GridSearchCV
            results = train_with_gridsearch(X_train, X_test, y_train, y_test, pipeline, param_grid, X_eval, y_eval)
            
            # Step 6: Save best model
            model_metadata = save_gridsearch_model(results, preprocessing_metadata)
            
            # Step 7: Conditionally register model
            register_if_best_model(results, model_metadata["model_path"])
            
            print("=" * 60)
            print("TRAINING COMPLETED SUCCESSFULLY!")
            print(f"Best F1 Score: {results['test_f1_score']:.4f}")
            print(f"Best Parameters: {results['best_params']}")
            print("=" * 60)
            
            # Log preprocessing metadata
            mlflow.log_params({
                "n_samples": len(X_train) + len(X_test),
                "n_features": 1 if isinstance(X_train, pd.Series) else X_train.shape[1],
                "text_version": "text_classical"
            })

        # mlflow.end_run()
            
        except Exception as e:
            print(f"Error during training: {e}")
            raise

if __name__ == "__main__":
    main()


