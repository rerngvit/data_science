# Credit: adapt from https://github.com/big-data-europe/docker-spark/blob/master/master/Dockerfile

FROM bde2020/spark-base:2.4.0-hadoop2.7

COPY master.sh /

ENV SPARK_MASTER_PORT 7077
ENV SPARK_MASTER_WEBUI_PORT 8080
ENV SPARK_MASTER_LOG /spark/logs

EXPOSE 8080 7077 6066

CMD ["/bin/bash", "/master.sh"]