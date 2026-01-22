FROM apache/airflow:3.1.6

USER airflow
RUN pip install --no-cache-dir \
    apache-airflow-providers-common-compat \
    apache-airflow-providers-common-messaging \
    apache-airflow-providers-common-sql \
    apache-airflow-providers-neo4j \
    apache-airflow-providers-postgres \
    apache-airflow-providers-pgvector \
    apache-airflow-providers-qdrant \
    apache-airflow-providers-redis \
    --constraint "${HOME}/constraints.txt" \
