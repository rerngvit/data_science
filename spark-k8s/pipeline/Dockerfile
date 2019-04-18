# Credit: adapt from https://github.com/big-data-europe/docker-spark/blob/master/master/Dockerfile

FROM rerng007/spark-standalone-k8s:spark-base

RUN apt-get update 

# Basic Ubuntu Dependencies  
RUN apt-get install -y curl python3 python3-pip python-dev build-essential git-core zip \
                 python3-dev python3-setuptools 

# Network utilities tool
RUN apt-get install -y wget dnsutils iputils-ping

# Jupyter
RUN pip3 install --upgrade pip setuptools
RUN pip3 install jupyter
RUN pip3 install jupyterlab
RUN pip3 install ipykernel && python3 -m ipykernel install  

# Python packages
RUN pip3 install virtualenv 

# Workflow processing for batch jobs
RUN pip3 install luigi

ENV SLUGIFY_USES_TEXT_UNIDECODE=yes
RUN pip3 install apache-airflow

# General data processing
RUN pip3 install pandas numpy 

# Machine learning
RUN pip3 install sklearn tensorflow imblearn keras GPy tpot fancyimpute tensorflow-hub

# Data access library
RUN pip3 install petastorm pyarrow

# Cloud API access
RUN pip3 install boto3 awscli

# Natural language processing
RUN pip3 install nltk gensim praw

# Download NLTK Data
# RUN python3 -m nltk.downloader -d /usr/local/share/nltk_data all

COPY kube_config /

# For jupyter lab
ENV PORT 8888
EXPOSE $PORT

# For spark bin
ENV SPARK_HOME /spark
ENV PATH $PATH:$SPARK_HOME/bin

# For other use cases
EXPOSE 8080 8082 9000

RUN mkdir /data
CMD cd /data && \
     /usr/local/bin/jupyter lab --port=$PORT --ip=0.0.0.0 --no-browser --allow-root
