FROM pytorch/pytorch
RUN apt-get install screen python-opencv -y
RUN conda install jupyter -y