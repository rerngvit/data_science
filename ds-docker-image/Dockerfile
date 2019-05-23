FROM ubuntu:18.04

ENV PORT 8888
EXPOSE $PORT

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

# General data processing
RUN pip3 install pandas numpy pyspark pyarrow


# Workflow processing for batch jobs
RUN pip3 install luigi

ENV SLUGIFY_USES_TEXT_UNIDECODE=yes
RUN pip3 install apache-airflow

# Machine learning
RUN pip3 install sklearn tensorflow imblearn keras GPy tpot fancyimpute

# Probablistic programming
RUN pip3 install edward

# Statistical computation
RUN pip3 install scipy bootstrapped

# Visualization
RUN pip3 install bokeh wordcloud seaborn 

# Cloud API access
RUN pip3 install boto3 awscli

# Natural language processing
RUN pip3 install nltk gensim

# Download NLTK Data
RUN python3 -m nltk.downloader -d /usr/local/share/nltk_data all

# Dask packages requirement
RUN pip3 install h5py Pillow matplotlib scipy toolz snakeviz graphviz 
RUN pip3 install dask distributed 
 
RUN mkdir /data
CMD cd /data && \
     /usr/local/bin/jupyter lab --port=$PORT --ip=0.0.0.0 --no-browser --allow-root
