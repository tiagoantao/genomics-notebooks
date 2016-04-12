FROM andrewosh/binder-base

MAINTAINER Tiago Antao <tiagoantao@gmail.com>

USER root

USER main

# Install Julia kernel
RUN bash docker_prepare.sh
