FROM pytorch/pytorch
RUN apt-get install screen python-opencv -y
RUN conda install jupyter -y
RUN conda install pytorch torchvision cudatoolkit=10.0 -c pytorch -y
RUN pip install opencv-python
