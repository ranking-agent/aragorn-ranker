# This Dockerfile is used to build a ROBOKOP python image
# starts with the python image
# installs vim
# creates /home/murphy
# sets HOME=/home/murphy and USER=murphy

FROM python:3.7.4-buster

# install basic tools
RUN apt-get update
RUN apt-get install -yq \
    vim

# set up murphy
RUN mkdir /home/murphy
ENV HOME=/home/murphy
ENV USER=murphy

# install requirements
ADD ./requirements.txt /home/murphy/requirements.txt
RUN pip install -r /home/murphy/requirements.txt --src /usr/local/src

# finish
WORKDIR /home/murphy