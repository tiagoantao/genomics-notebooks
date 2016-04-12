FROM andrewosh/binder-base

MAINTAINER Tiago Antao <tiagoantao@gmail.com>

USER root

USER main

# Install Julia kernel
RUN source activate python3; pip install simupop; source deactivate
