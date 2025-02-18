# 3D Scanning & Motion Capture (IN2354) - Development Environment

Docker image with all the required dependencies for the lecture https://niessner.github.io/3DScanning/


## Install Docker (and `docker-compose` or `docker compose`)

The first step is installing docker: https://docs.docker.com/engine/install/

## Build the development image

The following command will take some time. It will create a Docker image and install and compile the required dependencies.

```bash
cd Docker/
docker build . -t 3dsmc
```

## Create the container with Volume Mount
```bash
docker run -it --name 3dsmc -v "$(pwd):/workspace" 3dsmc /bin/bash
```

## Start the container with Volume Mount - Run this in the project directory. (Not in Docker folder)
```bash
docker start 3dsmc
docker exec -it -w /workspace 3dsmc /bin/bash
```






