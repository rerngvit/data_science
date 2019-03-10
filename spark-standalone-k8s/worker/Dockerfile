# Credit: adapt from https://github.com/big-data-europe/docker-spark/blob/master/worker/Dockerfile

FROM bde2020/spark-base:2.4.0-hadoop2.7

COPY worker.sh /

ENV SPARK_WORKER_WEBUI_PORT 8081
ENV SPARK_WORKER_LOG /spark/logs

EXPOSE 8081

CMD ["/bin/bash", "/worker.sh"]