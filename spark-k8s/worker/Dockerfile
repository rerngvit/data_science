# Credit: adapt from https://github.com/big-data-europe/docker-spark/blob/master/worker/Dockerfile

FROM rerng007/spark-standalone-k8s:spark-base

COPY worker.sh /

ENV SPARK_WORKER_LOG /spark/logs

EXPOSE 8081

CMD ["/wait-for-step.sh", "/worker.sh"]