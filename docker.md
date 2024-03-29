# Installing and deploying docker - Notes

## Creating a standalone docker image (.e. without docker compose)
Some definitions:

- Image : Blue print
- Container : Instantiation of a blueprint


`
docker build -t flaskapp:latest
`

`
docker run -it -d -p5000:5000 flaskapp
`
without -d for without daemon mode. To stop use ^C

To see running containers: docker ps

To run shell in a docker container
`
docker exec -it <jfjdhsfj3k4> /bin/sh 
`
The container id is from the previous command

`
docker stop <jfjdhsfj3k4>
`

## Docker compose build
[Put the volume file correctly - volume is loaded at only run time and not at build hence environment has to be copied]

```
docker-compose up -d

docker-compose stop
```

## Delete images
`
docker -images
        `
docker rmi $(script to delete all see in internet)
`j


## Generating environment files
`
conda env export --no-builds --from-history > environment.yml
`

Edit the above yml file to add - conda-forge before - defaults


To remove all images and containers
docker system prune -a
docker volume ls
docker volume prune
docker run -it --name aideal_container_dev --net=host -v ~/Documents/aiDealv1/:/home/aiDeal aideal_dev
docker build -t aideal_dev -f src/Dockerfile src/
docker run -it --name aideal_container_dev -p 5000:5000 -v ~/Documents/aiDealv1/:/home/aiDeal aideal_dev
docker exec -it aideal_container_dev bash
docker ps -a -f status=running
docker images
