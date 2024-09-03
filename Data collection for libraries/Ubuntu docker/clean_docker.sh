#/bin/bash

docker stop $(docker ps -a -q)
docker rm $(docker ps -a -q)
docker images | grep testubuntu | awk '{print $3}' | xargs docker rmi
docker image prune -f
