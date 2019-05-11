# Credit: adapt from https://github.com/big-data-europe/docker-spark/blob/master/worker/Dockerfile

FROM rerng007/spark-standalone-k8s:spark-base

COPY worker.sh /

ENV SPARK_WORKER_LOG /spark/logs

EXPOSE 8081

# Basic Ubuntu Dependencies  
RUN apt-get install -y curl python3 python3-pip python-dev build-essential git-core zip \
                 python3-dev python3-setuptools 

# Network utilities tool
RUN apt-get install -y dnsutils iputils-ping

# General data processing
RUN pip3 install pandas numpy 

# Machine learning
RUN pip3 install sklearn imblearn keras GPy tpot fancyimpute

# Tensorflow custom version to support CPU without AVX instruction set
RUN pip3 install https://github.com/rerngvit/tensorflow-prebuilt-binaries/blob/master/Linux/1.12.2/py36/tensorflow-1.12.2-cp36-cp36m-linux_x86_64.whl?raw=true
RUN pip3 install tensorflow-hub

# Data access library
RUN pip3 install petastorm pyarrow

# Natural language processing
RUN pip3 install nltk gensim praw

# Download NLTK Data
RUN python3 -m nltk.downloader -d /usr/local/share/nltk_data all


CMD ["/wait-for-step.sh", "/worker.sh"]