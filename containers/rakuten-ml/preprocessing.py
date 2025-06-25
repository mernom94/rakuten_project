#!/usr/bin/env python3
"""
Data preprocessing script for Rakuten product classification
Handles feature extraction and data preparation for ML training
Based on proven approach from previous project
"""

import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import os
from datetime import datetime
import json
import string
import nltk
from nltk.corpus import stopwords

# Download required NLTK data
nltk.download('stopwords', quiet=True)

# Database connection
DATABASE_URL = "postgresql+psycopg2://rakutenadmin:rakutenadmin@postgres:5432/rakuten_db"
engine = create_engine(DATABASE_URL)

# Output directories
PROCESSED_DATA_DIR = 'processed_data'
MODELS_DIR = 'models'

def create_output_dirs():
    """Create necessary output directories"""
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

def load_raw_data():
    """Load raw data from PostgreSQL database"""
    print("Loading raw data from PostgreSQL...")
    
    # Load features (text data)
    features_query = 'SELECT id, designation, description FROM "x_train"'
    X_data = pd.read_sql(features_query, con=engine)
    
    # Load targets
    targets_query = 'SELECT id, prdtypecode FROM "y_train"'
    y_data = pd.read_sql(targets_query, con=engine)
    
    # Merge on id
    data = pd.merge(X_data, y_data, on='id')
    
    print(f"Loaded {len(data)} samples with {data['prdtypecode'].nunique()} unique categories")
    return data

def preprocess_text(text):
    """
    Preprocess text: lowercase, remove punctuation, and remove stop words
    Based on the proven approach from previous project
    """
    if pd.isna(text):  # Handle NaN values
        return ""
    
    text = text.lower()  # Convert to lowercase, because stopwords are case-sensitive
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    
    # Get French stop words (matching original approach)
    french_stop_words = set(stopwords.words('french'))
    text = ' '.join([word for word in text.split() if word not in french_stop_words])  # Remove stop words
    
    return text

def create_processed_dataframe(data):
    """
    Create processed DataFrame with multiple text versions:
    - text_raw: Combined but minimal cleaning
    - text_classical: Heavy preprocessing for classical ML (current approach)
    - text_bert: Placeholder for future BERT preprocessing
    """
    print("Creating processed DataFrame with multiple text versions...")
    
    # Fill NaN values with empty string
    data['designation'] = data['designation'].fillna('')
    data['description'] = data['description'].fillna('')
    
    # Create a copy for processing
    df_processed = data[['id', 'designation', 'description', 'prdtypecode']].copy()
    
    # 1. Create raw combined text (minimal cleaning)
    print("Creating text_raw (minimal cleaning)...")
    df_processed['text_raw'] = (df_processed['designation'] + ' ' + df_processed['description']).str.strip()
    
    # 2. Create classical ML text (heavy preprocessing)
    print("Creating text_classical (French stopwords, punctuation removal)...")
    df_processed['text_classical'] = df_processed['text_raw'].apply(preprocess_text)
    
    # 3. Placeholder for future BERT text (minimal preprocessing)
    print("Creating text_bert placeholder (basic cleaning only)...")
    df_processed['text_bert'] = df_processed['text_raw'].str.lower().str.strip()
    
    # Keep all text versions and target
    df_processed = df_processed[['id', 'text_raw', 'text_classical', 'text_bert', 'prdtypecode']]
    
    # Remove rows with empty classical text (since that's what we're using now)
    df_processed = df_processed[df_processed['text_classical'].str.len() > 0]
    
    print(f"After preprocessing: {len(df_processed)} samples remain")
    print(f"Sample text_raw: {df_processed['text_raw'].iloc[0][:100]}...")
    print(f"Sample text_classical: {df_processed['text_classical'].iloc[0][:100]}...")
    
    return df_processed

def extract_text_features(df_processed, max_features=1000):
    """Extract TF-IDF features from classical ML preprocessed text"""
    print("Extracting TF-IDF features from text_classical...")
    
    # Initialize TF-IDF vectorizer (keeping parameters similar to original approach)
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),    # Unigrams and bigrams
        min_df=2,              # Ignore terms that appear in less than 2 documents
        max_df=0.8,            # Ignore terms that appear in more than 80% of documents
        lowercase=False,       # Already lowercased in preprocessing
        strip_accents='unicode'
    )
    
    # Fit and transform the classical ML text data
    X_features = vectorizer.fit_transform(df_processed['text_classical'])
    
    print(f"Created {X_features.shape[1]} TF-IDF features")
    
    return X_features, vectorizer

def save_processed_data(X_features, y_target, vectorizer, df_processed, data_info):
    """Save processed features, text versions, and vectorizer"""
    print("Saving processed data...")
    
    def convert_numpy_types(obj):
        """Convert numpy types to Python native types for JSON serialization"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        return obj

    # Create timestamp for versioning
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save features (sparse matrix)
    features_path = os.path.join(PROCESSED_DATA_DIR, f'X_features_{timestamp}.npz')
    from scipy.sparse import save_npz
    save_npz(features_path, X_features)
    
    # Save targets
    targets_path = os.path.join(PROCESSED_DATA_DIR, f'y_target_{timestamp}.npy')
    np.save(targets_path, y_target.values)
    
    # Save processed text DataFrame (with all text versions)
    text_path = os.path.join(PROCESSED_DATA_DIR, f'processed_text_{timestamp}.csv')
    df_processed.to_csv(text_path, index=False)
    
    # Save vectorizer
    vectorizer_path = os.path.join(MODELS_DIR, f'vectorizer_{timestamp}.pkl')
    with open(vectorizer_path, 'wb') as f:
        pickle.dump(vectorizer, f)
    
    # Save preprocessing metadata
    metadata = {
        'timestamp': timestamp,
        'features_path': features_path,
        'targets_path': targets_path,
        'text_path': text_path,
        'vectorizer_path': vectorizer_path,
        'n_samples': X_features.shape[0],
        'n_features': X_features.shape[1],
        'n_categories': len(np.unique(y_target)),
        'preprocessing_approach': 'multi_text_versions',
        'text_versions': {
            'text_raw': 'Combined designation + description, minimal cleaning',
            'text_classical': 'French stopwords + punctuation removal for classical ML',
            'text_bert': 'Basic cleaning only (placeholder for future BERT)'
        },
        'feature_extraction_params': {
            'max_features': vectorizer.max_features,
            'ngram_range': vectorizer.ngram_range,
            'min_df': vectorizer.min_df,
            'max_df': vectorizer.max_df,
            'source_text': 'text_classical'
        },
        'data_info': data_info
    }
    
    metadata_path = os.path.join(PROCESSED_DATA_DIR, f'preprocessing_metadata_{timestamp}.json')
    with open(metadata_path, 'w') as f:
        json.dump(convert_numpy_types(metadata), f, indent=2)
    
    # Save "latest" symlinks for easy access
    latest_metadata_path = os.path.join(PROCESSED_DATA_DIR, 'latest_preprocessing.json')
    with open(latest_metadata_path, 'w') as f:
        json.dump(convert_numpy_types(metadata), f, indent=2)

    print(f"Features saved: {features_path}")
    print(f"Targets saved: {targets_path}")
    print(f"Processed text saved: {text_path}")
    print(f"Vectorizer saved: {vectorizer_path}")
    print(f"Metadata saved: {metadata_path}")
    
    return metadata

def get_data_statistics(df_processed):
    """Generate basic statistics about the dataset"""
    stats = {
        'total_samples': len(df_processed),
        'unique_categories': df_processed['prdtypecode'].nunique(),
        'category_distribution': df_processed['prdtypecode'].value_counts().to_dict(),
        'text_stats': {
            'avg_raw_length': df_processed['text_raw'].str.len().mean(),
            'avg_classical_length': df_processed['text_classical'].str.len().mean(),
            'avg_bert_length': df_processed['text_bert'].str.len().mean(),
            'empty_classical_count': (df_processed['text_classical'].str.len() == 0).sum()
        }
    }
    return stats

def main():
    """Main preprocessing pipeline"""
    print("Starting data preprocessing pipeline...")
    print("Using French stopwords and punctuation removal (matching original approach)")
    print("=" * 60)
    
    try:
        # Step 1: Create output directories
        create_output_dirs()
        
        # Step 2: Load raw data
        data = load_raw_data()
        
        # Step 3: Create processed DataFrame (matching original approach)
        df_processed = create_processed_dataframe(data)
        
        # Step 4: Get data statistics
        data_stats = get_data_statistics(df_processed)
        print(f"Dataset statistics: {data_stats}")
        
        # Step 5: Extract features
        X_features, vectorizer = extract_text_features(df_processed)
        y_target = df_processed['prdtypecode']
        
        # Step 6: Save processed data
        metadata = save_processed_data(X_features, y_target, vectorizer, df_processed, data_stats)
        
        print("=" * 60)
        print("Preprocessing completed successfully!")
        print(f"Processed {metadata['n_samples']} samples")
        print(f"Generated {metadata['n_features']} features")
        print(f"Ready for training with {metadata['n_categories']} categories")
        
        return metadata
        
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        raise

if __name__ == "__main__":
    main()
