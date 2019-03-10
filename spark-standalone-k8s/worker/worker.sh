#!/bin/bash

. "/spark/sbin/spark-config.sh"

. "/spark/bin/load-spark-env.sh"

mkdir -p $SPARK_WORKER_LOG

export SPARK_HOME=/spark

ln -sf /dev/stdout $SPARK_WORKER_LOG/spark-worker.out

echo "Starting spark worker for host at '$SPARK_MASTER_HOST' and port at '$SPARK_MASTER_PORT' "

/spark/sbin/../bin/spark-class org.apache.spark.deploy.worker.Worker spark://$SPARK_MASTER_HOST:7077 >> $SPARK_WORKER_LOG/spark-worker.out