from datetime import datetime, timedelta
from airflow import DAG
from airflow.decorators import task
import logging

default_args = {
    "owner": "airflow",
    "retries": 0,
    "retry_delay": timedelta(minutes=1),
}

with DAG(
    dag_id="test_statsd_metrics",
    default_args=default_args,
    description="Emit test StatsD metrics from Airflow",
    start_date=datetime(2025, 7, 8),
    schedule=None,
    catchup=False,
    tags=["test", "metrics"],
) as dag:

    @task
    def emit_fake_metrics():
        import statsd
        import time

        logger = logging.getLogger("airflow.task")

        try:
            logger.info("Connecting to statsd exporter...")
            client = statsd.StatsClient(
                host="statsd-exporter",
                port=8125,
                prefix="airflow.test"
            )

            logger.info("Sending counter metric...")
            client.incr("fake_task_runs")

            logger.info("Sending timing metric...")
            client.timing("fake_task_duration", 250)  # 250ms

            logger.info("Sending gauge metric...")
            client.gauge("fake_task_value", 42)

            time.sleep(1)  # Give time for the exporter to process

            logger.info("All test metrics sent successfully.")

        except Exception as e:
            logger.error(f"Failed to send metrics: {e}")
            raise

    emit_fake_metrics()

