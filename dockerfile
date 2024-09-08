# Builds from our version of debian
FROM debian:bullseye

# install packages
RUN apt-get update && apt-get install -y \
	python3 \
	wget \
	libgraphite2-dev \
	imagemagick \
	python3-pip

# use that to install pip 
RUN pip install pipenv

# add in all the cool stuff

# our paper
ADD paper /opt/paper

# tectonic
RUN cd /opt/paper && wget 'https://github.com/tectonic-typesetting/tectonic/releases/download/tectonic%400.14.1/tectonic-0.14.1-x86_64-unknown-linux-gnu.tar.gz' && tar -xzf *.tar.gz && rm *.tar.gz && chmod +x tectonic

# pre-cache tectonic to pull in all the downloads
RUN cd /opt/paper && ./tectonic main.tex && rm ./main.pdf


# gurobi (minus the license file and compressed version)
ADD gurobi1003/linux64 /opt/gurobi1003/linux64
# and the example environment file
ADD gurobi1003/example_env /opt/gurobi1003/example_env

# our code
ADD FairDiversityandClusteringTemp /opt/FairDiversityandClusteringTemp
WORKDIR /opt/FairDiversityandClusteringTemp/

# remove the environment file from our normal setup (the user will supply it)
# alongside anything else we don't want
# and then make an environment file that just grabs the one in IO
RUN rm -rf .env gurobi.log ./publish/setup/result_* \
	&& ln -s /opt/IO/env .env


# install our virtual environment dependencies
RUN pipenv install

# IO directory
RUN mkdir /opt/IO/

# finally add in our reproducibility script
ADD reproducibility.sh /opt/FairDiversityandClusteringTemp/reproducibility.sh

# Sets up running our script in the docker container
ENTRYPOINT [ "pipenv", "run", "bash", "reproducibility.sh" ]
