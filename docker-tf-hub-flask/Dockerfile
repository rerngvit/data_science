FROM ubuntu:18.04

ENV PORT 9090
EXPOSE $PORT

RUN apt-get update 

# Basic Ubuntu Dependencies  
RUN apt-get install -y curl python3 python3-pip python-dev build-essential \
                 python3-dev python3-setuptools
                
# Basic Ubuntu Dependencies  
RUN apt-get install -y curl python3 python3-pip python-dev build-essential git-core zip \
                 python3-dev python3-setuptools 

RUN pip3 install --upgrade pip setuptools

# Network utilities tool
RUN apt-get install -y iputils-ping

# General data processing
RUN pip3 install pandas numpy 

# Tensorflow custom version to support CPU without AVX instruction set
RUN pip3 install https://github.com/rerngvit/tensorflow-prebuilt-binaries/blob/master/Linux/1.12.2/py36/tensorflow-1.12.2-cp36-cp36m-linux_x86_64.whl?raw=true
RUN pip3 install tensorflow-hub

# Flask
RUN pip3 install flask flask_restful

COPY src /src

ENTRYPOINT ["python"]

CMD ["src/app.py"]