FROM jupyter/notebook
MAINTAINER Tiago Antao <tra@popgen.net>

ENV DEBIAN_FRONTEND noninteractive   

RUN apt-get update
RUN apt-get install --force-yes -y libfreetype6-dev libpng12-dev
RUN pip install matplotlib
RUN apt-get install --force-yes -y liblapack3 libgfortran3 libblas3
RUN apt-get install --force-yes -y liblapack-dev libblas-dev
RUN apt-get install --force-yes -y gfortran
RUN pip install scipy
RUN pip install scikit-learn
RUN apt-get install --force-yes -y swig
RUN pip install simupop
RUN pip install networkx
WORKDIR /
RUN git clone https://github.com/tiagoantao/genomics-notebooks.git
WORKDIR /genomics-notebooks
CMD jupyter notebook --no-browser --ip=*
