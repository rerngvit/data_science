FROM ubuntu:18.04

ENV PORT 8888 8080
EXPOSE $PORT


ENV ENABLE_INIT_DAEMON true
ENV INIT_DAEMON_BASE_URI http://identifier/init-daemon
ENV INIT_DAEMON_STEP spark_master_init

ENV SPARK_VERSION=2.4.0
ENV HADOOP_VERSION=2.7

COPY wait-for-step.sh /
COPY execute-step.sh /
COPY finish-step.sh /

RUN apt-get update

# Java and Python
RUN apt-get install -y openjdk-8-jdk python3.6 python3-pip 

# Network utilities tool
RUN apt-get install -y wget dnsutils iputils-ping

# Vi editor
RUN apt-get install -y vim 

# Version control with git
RUN apt-get install -y git-core 

# Zip utility
RUN apt-get install -y zip

RUN   chmod +x *.sh \
      && wget http://apache.mirror.iphh.net/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz \
      && tar -xvzf spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz \
      && mv spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION} spark \
      && rm spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz \
      && cd /

ENV PYTHONHASHSEED 1