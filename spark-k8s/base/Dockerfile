#FROM ubuntu:18.04
FROM puckel/docker-airflow

ENV PORT 8888 8080
EXPOSE $PORT

ENV ENABLE_INIT_DAEMON true
ENV INIT_DAEMON_BASE_URI http://identifier/init-daemon
ENV INIT_DAEMON_STEP spark_master_init

ENV SPARK_VERSION=2.4.4
ENV HADOOP_VERSION=2.7

COPY wait-for-step.sh /
COPY execute-step.sh /
COPY finish-step.sh /

RUN apt-get update

# Java 
RUN apt-get install -y openjdk-8-jdk 

# Python
RUN apt-get install -y python3.6 python3-pip 

# Network utilities tool
RUN apt-get install -y wget dnsutils iputils-ping

# Vi editor
RUN apt-get install -y vim 

# Version control with git
RUN apt-get install -y git-core 

# Zip and Bzip2 utilities
RUN apt-get install -y zip bzip2

# Background execution utility
RUN apt-get install -y screen

# Postgresql client
RUN apt-get install -y postgresql-client-10
 
ARG SPARK_BINARY_URL=https://archive.apache.org/dist/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz
ARG SPARK_BINARY_FILE_NAME=spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}


RUN   chmod +x *.sh \
      && wget $SPARK_BINARY_URL \
      && tar -xvzf $SPARK_BINARY_FILE_NAME.tgz \
      && mv ${SPARK_BINARY_FILE_NAME} spark \
      && rm ${SPARK_BINARY_FILE_NAME}.tgz \
      && cd /

ENV PYTHONHASHSEED 1