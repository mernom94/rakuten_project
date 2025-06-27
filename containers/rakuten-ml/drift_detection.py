import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset, RegressionPreset
import os
from datetime import datetime



# dags/tasks/drift_detection.py

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
reports_dir = '/opt/airflow/reports'
os.makedirs(reports_dir, exist_ok=True)



# === Load processed dataset ===
full_data = pd.read_csv("data/processed_data.csv")

# === Simulate 'reference' and 'current' datasets, as i understood we wanted to 'simulate' new data coming in, maybe change this?===
reference_data = full_data.iloc[:500].copy()
current_data = full_data.iloc[500:1000].copy()

# Optional: save these slices for inspection
reference_data.to_csv("data/reference_data.csv", index=False)
current_data.to_csv("data/current_data.csv", index=False)

# === Create Evidently reports ===
data_drift_report = Report(metrics=[DataDriftPreset()])
target_drift_report = Report(metrics=[TargetDriftPreset(target="target")])
regression_report = Report(metrics=[RegressionPreset(target="target", prediction="prediction")])

# === Run reports ===
data_drift_report.run(reference_data=reference_data, current_data=current_data)
target_drift_report.run(reference_data=reference_data, current_data=current_data)
regression_report.run(reference_data=reference_data, current_data=current_data)

# === Save reports ===
data_drift_report.save_html(f"reports/data_drift_report_{timestamp}.html")
target_drift_report.save_html(f"reports/target_drift_report_{timestamp}.html")
regression_report.save_html(f"reports/regression_performance_report_{timestamp}.html")


