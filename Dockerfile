# leverage the renci python base image
FROM renciorg/renci-python-image:v0.0.1

#Build from this branch.  Default to master
ARG BRANCH_NAME=master

# make a directory for the repo
RUN mkdir /repo

# go to the directory where we are going to upload the repo
WORKDIR /repo

# get the latest code
RUN git clone --branch $BRANCH_NAME --single-branch https://github.com/ranking-agent/aragorn-ranker.git

# go to the repo dir
WORKDIR /repo/aragorn-ranker

# make sure all is writeable for the nru USER later on
RUN chmod -R 777 .

# install requirements
RUN pip install -r requirements.txt

# switch to the non-root user (nru). defined in the base image
USER nru

# expose the default port
EXPOSE 4868

# start the service entry point
ENTRYPOINT ["bash", "main.sh"]