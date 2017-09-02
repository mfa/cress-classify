#!/bin/sh
nvidia-docker build -f Dockerfile.gpu --tag cress_classify .
nvidia-docker run -d --rm -it -p 8888:8888 -v `pwd`/notebook:/notebooks --name cress_gpu cress_classify
nvidia-docker run -d --rm -it -p 6006:6006 -v `pwd`/notebook:/notebooks --entrypoint tensorboard cress_classify --logdir=/notebooks/logs/

