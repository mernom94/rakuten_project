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

# Directories
PROCESSED_DATA_DIR = 'processed_data'
MODELS_DIR = 'models'

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

def train_with_gridsearch(X_train, X_test, y_train, y_test, pipeline, param_grid):
    """Train models using GridSearchCV"""
    print("Starting GridSearchCV training...")
    print("This may take some time depending on the parameter grid size...")
    
    # Encode the labels
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)
    
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
    
    # Prepare results
    results = {
        'best_params': grid_search.best_params_,
        'best_cv_score': grid_search.best_score_,
        'test_f1_score': test_f1,
        'test_accuracy': test_accuracy,
        'label_encoder': label_encoder,
        'best_estimator': grid_search.best_estimator_
    }
    
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
        'best_params': results['best_params'],
        'performance': {
            'best_cv_score': results['best_cv_score'],
            'test_f1_score': results['test_f1_score'],
            'test_accuracy': results['test_accuracy']
        },
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

def main():
    """Main training pipeline with GridSearchCV"""
    print("Starting GridSearchCV training pipeline...")
    print("Based on proven approach from previous project")
    print("=" * 60)
    
    try:
        # Step 1: Load preprocessed data
        text_df, preprocessing_metadata = load_latest_processed_data()
        
        # Step 2: Extract features and target (using classical ML text)
        X = text_df['text_classical']  # Use heavily preprocessed text
        y = text_df['prdtypecode']
        
        print(f"Using text_classical for training ({len(X)} samples)")
        
        # Step 3: Create stratified train/test split
        X_train, X_test, y_train, y_test = create_stratified_split(X, y)
        
        # Step 4: Set up pipeline and parameter grid
        pipeline, param_grid = create_pipeline_and_param_grid()
        
        # Step 5: Train with GridSearchCV
        results = train_with_gridsearch(X_train, X_test, y_train, y_test, pipeline, param_grid)
        
        # Step 6: Save best model
        model_metadata = save_gridsearch_model(results, preprocessing_metadata)
        
        print("=" * 60)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print(f"Best F1 Score: {results['test_f1_score']:.4f}")
        print(f"Best Parameters: {results['best_params']}")
        print("=" * 60)
        
    except Exception as e:
        print(f"Error during training: {e}")
        raise

if __name__ == "__main__":
    main()


