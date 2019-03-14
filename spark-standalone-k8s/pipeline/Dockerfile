# Credit: adapt from https://github.com/big-data-europe/docker-spark/blob/master/master/Dockerfile

FROM bde2020/spark-base:2.4.0-hadoop2.7

RUN apt-get update 

# Basic Ubuntu Dependencies  
RUN apt-get install -y curl python3 python3-pip python-dev build-essential \
                 python3-dev python3-setuptools
                
# Jupyter
RUN pip3 install --upgrade pip setuptools
RUN pip3 install jupyter
RUN pip3 install jupyterlab
RUN pip3 install ipykernel && python3 -m ipykernel install  

# Python packages
RUN pip3 install virtualenv 

# Workflow processing
RUN luigi apache-airflow

# General data processing
RUN pip3 install pandas numpy 

# Machine learning
RUN pip3 install sklearn tensorflow imblearn keras GPy tpot fancyimpute

# Cloud API access
RUN pip3 install boto3 awscli

# Natural language processing
RUN pip3 install nltk gensim

# Download NLTK Data
RUN python3 -m nltk.downloader -d /usr/local/share/nltk_data all

EXPOSE 8080 8082 9000


CMD ["/bin/bash"]