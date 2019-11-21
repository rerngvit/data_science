# Credit: adapt from https://github.com/big-data-europe/docker-spark/blob/master/worker/Dockerfile

FROM rerng007/spark-native-k8s:base

COPY worker.sh /

ENV SPARK_WORKER_LOG /spark/logs

EXPOSE 8081

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