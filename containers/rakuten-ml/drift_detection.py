import os
import glob
import json
import pickle
import pandas as pd
import numpy as np
from evidently import Report
from evidently.presets import DataDriftPreset
from sklearn.preprocessing import LabelEncoder
import os
import json
import sys

# Helper to get the latest processed reference data path dynamically
def get_latest_reference_path(processed_data_dir='processed_data'):
    latest_meta_path = os.path.join(processed_data_dir, 'latest_preprocessing.json')
    if not os.path.exists(latest_meta_path):
        return None
    with open(latest_meta_path, 'r') as f:
        metadata = json.load(f)
    ref_path = metadata.get('text_path')  # CSV with processed text
    
    # First check if ref_path as is exists (could be absolute)
    if ref_path and os.path.exists(ref_path):
        return ref_path
    
    # Otherwise, check relative to processed_data_dir
    if ref_path:
        abs_ref_path = os.path.abspath(os.path.join(processed_data_dir, ref_path))
        if os.path.exists(abs_ref_path):
            return abs_ref_path
    
    return None


# TESTING MODE: If no data/train_reference.csv exists, generate a synthetic one using Iris
def generate_iris_test_data():
    from sklearn.datasets import load_iris
    df = load_iris(as_frame=True).frame
    df.columns = [col.replace(' (cm)', '').replace(' ', '_') for col in df.columns]  # clean column names
    os.makedirs('./data', exist_ok=True)
    df.to_csv('./data/train_reference.csv', index=False)
    print("Iris test data generated at ./data/train_reference.csv")

# Load model + label encoder
def load_latest_model_and_encoder():
    model_files = glob.glob('./models/best_model_gridsearch_*.pkl')
    encoder_files = glob.glob('./models/label_encoder_*.pkl')

    if not model_files or not encoder_files:
        raise FileNotFoundError("Model or encoder files not found")

    latest_model = sorted(model_files)[-1]
    latest_encoder = sorted(encoder_files)[-1]

    with open(latest_model, 'rb') as f:
        model = pickle.load(f)

    with open(latest_encoder, 'rb') as f:
        label_encoder = pickle.load(f)

    return model, label_encoder

# Load category mapping
def load_category_mapping():
    mapping_path = './containers/rakuten-ml/category_mapping.json'
    if os.path.exists(mapping_path):
        with open(mapping_path, 'r') as f:
            return json.load(f)
    return {}

# Determine categorical columns automatically (object dtype or low unique ratio)
def get_categorical_columns(df, threshold=0.05):
    return [
        col for col in df.columns
        if df[col].dtype == 'object' or (df[col].nunique() / len(df)) < threshold
    ]

# Simulate incoming chunk (random 5% sample)
def get_data_chunk(full_df, chunk_size=0.05):
    return full_df.sample(frac=chunk_size, random_state=np.random.randint(1_000_000))

# Run drift detection using Evidently
def run_drift_detection(ref_df, new_df, cat_cols, report_output="./drift_report.html"):

    report = Report(metrics=[DataDriftPreset()])
    report_to_save = report.run(reference_data=ref_df, current_data=new_df)

    report_to_save.save_html(report_output)

    drift_metrics = report_to_save.dict()
    print(json.dumps(drift_metrics, indent=2))
    
    # Find DriftedColumnsCount metric entry
    drift_count_metric = next(
        (m for m in drift_metrics['metrics'] if m['metric_id'].startswith('DriftedColumnsCount')), None
    )

    if drift_count_metric is None:
        raise ValueError("Could not find 'DriftedColumnsCount' metric in the report output")

    n_drifted = drift_count_metric['value']['count']  # number of drifted columns
    drift_share = drift_count_metric['value']['share']  # share of drifted columns

    total_cols = ref_df.shape[1]

    print(f"Drift detected in {n_drifted} of {total_cols} columns ({drift_share:.1%})")
    return drift_share > 0.3  # threshold for "significant" drift

def main():
    print("Loading model and encoder...")
    model, label_encoder = load_latest_model_and_encoder()

    print("Loading reference data...")
    reference_path = get_latest_reference_path()
    if reference_path is None or not os.path.exists(reference_path):
        print("No reference data found — generating Iris fallback test set.")
        generate_iris_test_data()
        reference_path = './data/train_reference.csv'

    print(f"Using reference data from: {reference_path}")
    ref_df = pd.read_csv(reference_path)

    print("Simulating new incoming data...")
    current_df = get_data_chunk(ref_df, chunk_size=0.05)

    cat_cols = get_categorical_columns(ref_df)
    print(f"Categorical columns detected: {cat_cols}")

    print("Running drift detection...")
    drift_detected = run_drift_detection(ref_df, current_df, cat_cols)

    if drift_detected:
        print("Significant data drift detected! Drift report saved to drift_report.html")
        # TODO: Add alerting integration here (Slack, email, etc.)
    else:
        print("No significant drift detected.")

def test_with_synthetic_model_and_data():
    from sklearn.datasets import load_iris
    from sklearn.ensemble import RandomForestClassifier

    print("=== Running test with synthetic Iris model and data ===")
    os.makedirs('./models', exist_ok=True)
    os.makedirs('./data', exist_ok=True)

    # Load Iris data
    iris = load_iris(as_frame=True)
    df = iris.frame.copy()
    df.columns = [col.replace(' (cm)', '').replace(' ', '_') for col in df.columns]

    # Save reference data
    df.to_csv('./data/train_reference.csv', index=False)

    # Prepare features and labels
    X = df.drop(columns=['target'])
    y = df['target']

    # Train a simple model
    label_encoder = LabelEncoder()
    y_enc = label_encoder.fit_transform(y)

    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y_enc)

    # Save model and encoder
    with open('./models/best_model_gridsearch_iris.pkl', 'wb') as f:
        pickle.dump(model, f)

    with open('./models/label_encoder_iris.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)

    # Simulate new data (5% sample)
    current_df = df.sample(frac=0.05, random_state=42)

    cat_cols = get_categorical_columns(df)
    print(f"Categorical columns detected: {cat_cols}")

    # Run drift detection
    drift_detected = run_drift_detection(df, current_df, cat_cols, report_output="test_drift_report.html")

    if drift_detected:
        print("Test: Significant data drift detected!")
    else:
        print("Test: No significant drift detected.")

if __name__ == "__main__":
    # Uncomment to run main with saved models/data
    # main()

    # Run test case with synthetic Iris model and data
    test_with_synthetic_model_and_data()

    sys.exit(0)


    
