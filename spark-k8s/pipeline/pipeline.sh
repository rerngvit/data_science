#!/bin/bash

airflow initdb
airflow webserver -p 8080 &
airflow scheduler &

# Creating the DAG folder
mkdir /root/airflow/dags

export PORT=8888
cd /data && /usr/local/bin/jupyter lab --port=$PORT --ip=0.0.0.0 --no-browser --allow-root
