#!/usr/bin/env python3
"""
Single prediction script for Rakuten product classification
Reuses existing preprocessing pipeline for consistency
"""
import sys
import json
import pickle
import pandas as pd
import numpy as np
from preprocessing import preprocess_text
import os
import glob

def load_category_mapping():
    """Load category number to name mapping"""
    try:
        with open('containers/rakuten-ml/category_mapping.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print("Warning: category_mapping.json not found, using numeric categories")
        return {}

def load_latest_model_and_encoder():
    # """Load the most recent trained model and label encoder"""
    
    # # Find latest model files by timestamp
    # model_files = glob.glob('models/best_model_gridsearch_*.pkl')
    # encoder_files = glob.glob('models/label_encoder_*.pkl')
    
    # if not model_files or not encoder_files:
    #     raise FileNotFoundError("Missing model files. Make sure training has been completed.")
    
    # # Get the latest files (by timestamp in filename)
    # latest_model = sorted(model_files)[-1]
    # latest_encoder = sorted(encoder_files)[-1]
    
    latest_model = 'models/the_best_model.pkl'
    latest_encoder = 'models/the_label_encoder.pkl'
    
    print(f"Loading model: {latest_model}")
    print(f"Loading encoder: {latest_encoder}")
    
    # Load model (this is a Pipeline with vectorizer + classifier)
    with open(latest_model, 'rb') as f:
        model = pickle.load(f)
    
    # Load label encoder
    with open(latest_encoder, 'rb') as f:
        label_encoder = pickle.load(f)
    
    return model, label_encoder

def predict_single(title, description):
    """
    Make prediction for a single product using existing preprocessing pipeline
    """
    try:
        # Step 1: Load category mapping
        category_mapping = load_category_mapping()
        
        # Step 2: Combine title and description (same as preprocessing.py)
        combined_text = f"{title} {description}".strip()
        
        # Step 3: Apply same preprocessing as training
        text_classical = preprocess_text(combined_text)
        
        if not text_classical:
            return {
                "error": "Text became empty after preprocessing",
                "category": None,
                "confidence": 0.0,
                "top_3": []
            }
        
        print(f"Original text: {combined_text[:100]}...")
        print(f"Preprocessed text: {text_classical[:100]}...")
        
        # Step 4: Load model and encoder
        model, label_encoder = load_latest_model_and_encoder()
        
        # Step 5: The model is a pipeline that expects raw text (it handles vectorization internally)
        prediction_encoded = model.predict([text_classical])[0]
        prediction_numeric = label_encoder.inverse_transform([prediction_encoded])[0]
        
        # Convert numeric category to name
        prediction_category = category_mapping.get(str(prediction_numeric), f"Unknown Category {prediction_numeric}")
        
        print(f"Encoded prediction: {prediction_encoded}")
        print(f"Numeric prediction: {prediction_numeric}")
        print(f"Category name: {prediction_category}")
        
        # Step 6: Get prediction probabilities
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba([text_classical])[0]
            classes_encoded = model.classes_
            classes_numeric = label_encoder.inverse_transform(classes_encoded)
            
            # Get top 3 predictions
            top_3_indices = np.argsort(probabilities)[-3:][::-1]
            top_3 = [
                {
                    "category": category_mapping.get(str(classes_numeric[i]), f"Unknown Category {classes_numeric[i]}"),
                    "confidence": float(probabilities[i])
                }
                for i in top_3_indices
            ]
            
            # Find confidence for the top prediction
            confidence = float(probabilities[classes_encoded == prediction_encoded][0])
            
        else:
            # For models without predict_proba (like SVM without probability=True)
            confidence = 1.0
            top_3 = [{"category": prediction_category, "confidence": 1.0}]
        
        return {
            "category": prediction_category,
            "confidence": confidence,
            "top_3": top_3
        }
        
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        return {
            "error": str(e),
            "category": None,
            "confidence": 0.0,
            "top_3": []
        }

if __name__ == "__main__":
    # Accept input from command line as JSON
    if len(sys.argv) != 2:
        print("Usage: python predict.py '{\"title\": \"...\", \"description\": \"...\"}'")
        sys.exit(1)
    
    try:
        input_data = json.loads(sys.argv[1])
        result = predict_single(input_data["title"], input_data["description"])
        print(json.dumps(result))
    except json.JSONDecodeError as e:
        print(json.dumps({"error": f"Invalid JSON input: {str(e)}"}))
    except KeyError as e:
        print(json.dumps({"error": f"Missing required field: {str(e)}"}))
