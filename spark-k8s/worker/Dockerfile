# Credit: adapt from https://github.com/big-data-europe/docker-spark/blob/master/worker/Dockerfile

FROM rerng007/spark-standalone-k8s:spark-base

COPY worker.sh /

ENV SPARK_WORKER_LOG /spark/logs

EXPOSE 8081

# Basic Ubuntu Dependencies  
RUN apt-get install -y curl python3 python3-pip python-dev build-essential git-core zip \
                 python3-dev python3-setuptools 

# General data processing
RUN pip3 install pandas numpy 

# Machine learning
RUN pip3 install sklearn tensorflow imblearn keras GPy tpot fancyimpute tensorflow-hub

# Data access library
RUN pip3 install petastorm pyarrow

# Natural language processing
RUN pip3 install nltk gensim praw

# Download NLTK Data
RUN python3 -m nltk.downloader -d /usr/local/share/nltk_data all


CMD ["/wait-for-step.sh", "/worker.sh"]