import streamlit as st
import streamlit.components.v1 as components
from containers.rakuten_st.streamlit_utils import add_pagination_and_footer
import streamlit as st
import pandas as pd
from datetime import datetime
import requests

def query_prometheus(query):
    url = "http://localhost:9090/api/v1/query"
    resp = requests.get(url, params={"query": query})
    resp.raise_for_status()
    data = resp.json()
    return data["data"]["result"]

def parse_timeseries_result(result):
    # result example: [{"metric": {...}, "value": [timestamp, value]}]
    if not result:
        return None, None
    timestamps = []
    values = []
    for item in result:
        ts, val = item["value"]
        timestamps.append(pd.to_datetime(float(ts), unit='s'))
        values.append(float(val))
    return timestamps, values

st.set_page_config(
    page_title="MAY25 BDS // Monitoring and Operations",
    page_icon="images/logos/rakuten-favicon.ico",
    layout="wide",
)

st.progress(6 / 8)
st.title("Monitoring and Operations")

st.markdown("""
In this section, we monitor the health and performance of our ML pipeline and infrastructure.

We rely on:
- **Prometheus** for metrics scraping
- **Grafana** for real-time dashboards
- **FastAPI** with Prometheus instrumentation for API metrics
- **Node Exporter** for system-level resource monitoring

Below you’ll find live metrics embedded directly into this app.
""")



st.title("FastAPI Metrics")

# Total Requests in last hour
total_requests_data = query_prometheus('sum(increase(http_requests_total[1h]))')
total_requests = float(total_requests_data[0]['value'][1]) if total_requests_data else 0
st.metric("Total Requests (Last Hour)", f"{total_requests:.0f}")

# Requests per second (rate)
rps_data = query_prometheus('sum(rate(http_requests_total[1m]))')
rps = float(rps_data[0]['value'][1]) if rps_data else 0
st.metric("Requests per Second", f"{rps:.2f}")

# Requests In Progress
in_progress_data = query_prometheus('sum(http_requests_in_progress_total)')
in_progress = float(in_progress_data[0]['value'][1]) if in_progress_data else 0
st.metric("Requests In Progress", f"{in_progress:.0f}")

# Request latency P95 (this is a time series, but we can show latest value)
latency_data = query_prometheus('histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket[1m])) by (le))')
latency = float(latency_data[0]['value'][1]) if latency_data else 0
st.metric("Request Latency P95 (seconds)", f"{latency:.3f}")

# Average Request Duration over 1m
avg_duration_data = query_prometheus('rate(http_request_duration_seconds_sum[1m]) / rate(http_request_duration_seconds_count[1m])')
avg_duration = float(avg_duration_data[0]['value'][1]) if avg_duration_data else 0
st.metric("Average Request Duration (seconds)", f"{avg_duration:.3f}")

# Requests by Endpoint (handler) — bar chart
requests_by_handler = query_prometheus('sum(rate(http_requests_total[1m])) by (handler)')
if requests_by_handler:
    df = pd.DataFrame({
        "handler": [res['metric'].get('handler', 'unknown') for res in requests_by_handler],
        "rate": [float(res['value'][1]) for res in requests_by_handler]
    })
    df = df.sort_values("rate", ascending=False)
    st.subheader("Requests by Endpoint")
    st.bar_chart(df.set_index("handler"))

# Error Rate (4xx/5xx)
error_rate_data = query_prometheus('sum(rate(http_requests_total{status=~"4..|5.."}[1m]))')
error_rate = float(error_rate_data[0]['value'][1]) if error_rate_data else 0
st.metric("Error Rate (4xx/5xx per second)", f"{error_rate:.3f}")


from datetime import datetime, timedelta

def query_prometheus(query, start=None, end=None, step=None):
    url = "http://localhost:9090/api/v1/query_range" if start and end and step else "http://localhost:9090/api/v1/query"
    params = {"query": query}
    if start and end and step:
        params.update({
            "start": start.timestamp(),
            "end": end.timestamp(),
            "step": step
        })
    resp = requests.get(url, params=params)
    resp.raise_for_status()
    data = resp.json()
    return data["data"]["result"]

def parse_timeseries_result(result):
    if not result:
        return None, None
    timestamps = []
    values = []
    for item in result:
        if "values" in item:
            for ts, val in item["values"]:
                timestamps.append(pd.to_datetime(float(ts), unit='s'))
                values.append(float(val))
        elif "value" in item:
            ts, val = item["value"]
            timestamps.append(pd.to_datetime(float(ts), unit='s'))
            values.append(float(val))
    return timestamps, values

# Time range parameters
end = datetime.now()
start = end - timedelta(hours=24)
step = 300  # 5 minutes

st.title("Model Performance Metrics")

# Accuracy over time
accuracy_query = "mlflow_accuracy"
accuracy_data = query_prometheus(accuracy_query, start=start, end=end, step=step)
times, values = parse_timeseries_result(accuracy_data)

if times and values:
    st.subheader("Model Accuracy Over Time")
    df_acc = pd.DataFrame({"Time": times, "Accuracy": values})
    st.line_chart(df_acc.rename(columns={"Time": "index"}).set_index("index")["Accuracy"])

    best_acc_query = "max_over_time(mlflow_accuracy[24h])"
    best_acc_data = query_prometheus(best_acc_query)
    if best_acc_data:
        best_acc_val = float(best_acc_data[0]["value"][1])
        st.metric("Best Accuracy (Last 24h)", f"{best_acc_val:.2%}")

# Total Model Runs over time
runs_query = "mlflow_experiment_run_total"
runs_data = query_prometheus(runs_query, start=start, end=end, step=step)
times_runs, values_runs = parse_timeseries_result(runs_data)

if times_runs and values_runs:
    st.subheader("Total Model Runs Over Time")
    df_runs = pd.DataFrame({"Time": times_runs, "Runs": values_runs})
    st.line_chart(df_runs.rename(columns={"Time": "index"}).set_index("index")["Runs"])

# CV Score over time
cv_query = "mlflow_cv"
cv_data = query_prometheus(cv_query, start=start, end=end, step=step)
times_cv, values_cv = parse_timeseries_result(cv_data)

if times_cv and values_cv:
    st.subheader("CV Score Over Time")
    df_cv = pd.DataFrame({"Time": times_cv, "CV Score": values_cv})
    st.line_chart(df_cv.rename(columns={"Time": "index"}).set_index("index")["CV Score"])

    best_cv_query = "max_over_time(mlflow_cv[24h])"
    best_cv_data = query_prometheus(best_cv_query)
    if best_cv_data:
        best_cv_val = float(best_cv_data[0]["value"][1])
        st.metric("Best CV Score (Last 24h)", f"{best_cv_val:.2%}")

# F1 Score over time
f1_query = "mlflow_f1"
f1_data = query_prometheus(f1_query, start=start, end=end, step=step)
times_f1, values_f1 = parse_timeseries_result(f1_data)

if times_f1 and values_f1:
    st.subheader("F1 Score Over Time")
    df_f1 = pd.DataFrame({"Time": times_f1, "F1 Score": values_f1})
    st.line_chart(df_f1.rename(columns={"Time": "index"}).set_index("index")["F1 Score"])

    best_f1_query = "max_over_time(mlflow_f1[24h])"
    best_f1_data = query_prometheus(best_f1_query)
    if best_f1_data:
        best_f1_val = float(best_f1_data[0]["value"][1])
        st.metric("Best F1 Score (Last 24h)", f"{best_f1_val:.2%}")


import time 

# --- Node Exporter System Metrics Section ---

# Node Exporter Metrics
st.title("Node Exporter Metrics")

# Define node_exporter queries
node_exporter_queries = {
    "CPU Usage (avg over all cores) [%]": "100 - (avg by(instance) (rate(node_cpu_seconds_total{mode='idle'}[5m])) * 100)",
    "Memory Usage [%]": "100 * (1 - ((node_memory_MemAvailable_bytes) / node_memory_MemTotal_bytes))",
    "Disk Read Throughput [bytes/s]": "rate(node_disk_read_bytes_total[5m])",
    "Disk Write Throughput [bytes/s]": "rate(node_disk_written_bytes_total[5m])",
    "Network Receive Throughput [bytes/s]": "rate(node_network_receive_bytes_total[5m])",
    "Network Transmit Throughput [bytes/s]": "rate(node_network_transmit_bytes_total[5m])",
}

# Use a fixed range: last 1 hour, every 30 seconds
end = datetime.now()
start = end - timedelta(hours=1)
step = 30  # in seconds

def parse_multi_timeseries(result):
    # returns dict[label] = (timestamps, values)
    series = {}
    for item in result:
        labels = item.get("metric", {})
        label_parts = [f"{k}={v}" for k, v in sorted(labels.items()) if k != "__name__"]
        label = ", ".join(label_parts) if label_parts else "all"

        datapoints = item.get("values", [])
        timestamps = [pd.to_datetime(float(ts), unit="s") for ts, _ in datapoints]
        values = [float(val) for _, val in datapoints]
        series[label] = (timestamps, values)
    return series

for title, query in node_exporter_queries.items():
    try:
        result = query_prometheus(query, start=start, end=end, step=step)
        series = parse_multi_timeseries(result)

        if not series:
            st.warning(f"No data found for **{title}**")
            continue

        # Get a union of all timestamps
        all_timestamps = sorted(set(ts for ts_list, _ in series.values() for ts in ts_list))
        df = pd.DataFrame(index=all_timestamps)

        for label, (ts_list, vals) in series.items():
            s = pd.Series(data=vals, index=ts_list)
            df[label] = s

        df = df.sort_index()
        st.subheader(title)
        df.columns = [str(col).replace(":", "_") for col in df.columns]
        st.line_chart(df.fillna(method="ffill").fillna(method="bfill"))

    except Exception as e:
        st.error(f"Error fetching **{title}**: {e}")

# Auto refresh after 15s to mimic Grafana
st.write("Dashboard refreshes every 15 seconds.")
time.sleep(15)
st.rerun()

st.markdown("---")
add_pagination_and_footer("pages/6_Monitoring_and_Operations.py")

