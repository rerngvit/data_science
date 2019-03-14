# ds-docker-image
Minimalistic Docker image for small-to-medium data science tasks on Python stack

## How to use it
* Install Docker (https://docs.docker.com/install/)
* Run the following command on the terminal
  
  $ docker run -ti  -v "\`pwd\`":/data -p 8888:8888 rerng007/ds-docker-image:latest
