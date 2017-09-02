FROM jupyter/scipy-notebook
MAINTAINER xx

# get curl
USER root
RUN apt-get upgrade; apt-get update; apt-get install -y curl

# set user (the one from scipy-notebook base image)
USER jovyan

ENV KERAS_BACKEND tensorflow

ADD requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

WORKDIR /data
EXPOSE 8888
CMD jupyter notebook --no-browser --ip=0.0.0.0
