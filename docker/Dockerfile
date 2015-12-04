FROM jupyter/notebook
MAINTAINER Tiago Antao <tra@popgen.net>

RUN pip install matplotlib
RUN pip install scikit-learn
RUN pip install simupop
WORKDIR /
RUN git clone https://github.com/tiagoantao/genomics-notebooks.git
WORKDIR /genomics-notebooks
CMD jupyter notebook --no-browser --ip=*
