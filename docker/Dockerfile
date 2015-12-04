FROM jupyter/notebook
MAINTAINER Tiago Antao <tra@popgen.net>

WORKDIR /
RUN git clone https://github.com/tiagoantao/genomics-notebooks.git
WORKDIR /genomics-notebooks
CMD jupyter notebook --no-browser --ip=*
