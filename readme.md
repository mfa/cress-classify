# classify wet/dry cress box using CNNs on photos

## intro

- Data from https://cress.space/
- First public release with Talk on MRMCD17 (Talk in German) - https://cfp.mrmcd.net/2017/talk/EFZ97G/


## experiments

see experiments.org

## data

see data.org


## run

### run (cpu)

``docker-compose up``

now surf to: [http://localhost:8888/](http://localhost:8888/)

you want to use the GPU version (for a lot more speed)

### run (gpu)

you need: [nvidia-docker](https://github.com/NVIDIA/nvidia-docker)

```
./nvidia.sh
```

#### run sth inside the docker container

```
docker exec -ti cress_gpu /bin/bash
```

The easiest way is to prepare a shell script and run this inside:
```
sh run.sh
```


### run on fresh ubuntu zesty

```
apt install -y zip
curl -fsSL get.docker.com -o get-docker.sh && sh get-docker.sh
curl -L https://github.com/docker/compose/releases/download/1.14.0/docker-compose-`uname -s`-`uname -m` > /usr/local/bin/docker-compose
chmod a+x /usr/local/bin/docker-compose
# git clone ...
cd cress-classify
docker-compose build
docker-compose up -d
docker-compose logs main
```

