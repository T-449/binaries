#/bin/bash

docker build -t testubuntu .
docker run --privileged -it testubuntu
