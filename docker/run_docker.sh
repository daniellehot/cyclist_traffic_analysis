CODE_HOST=/home/daniel/cowi_cyclist_detection/
CODE_CONTAINER=/home/daniel/cowi_cyclist_detection/
#DATA_HOST=/home/daniel/OneDrive/autofish_groups/
#DATA_CONTAINER=/home/create.aau.dk/vo65hs/autofish_groups/
IMAGE=dale97/detectron2_container

xhost +local:docker
XSOCK=/tmp/.X11-unix
XAUTH=/tmp/.docker.xauth
docker run --gpus all -it --rm \
    -e DISPLAY=$DISPLAY \
    -v $XSOCK:$XSOCK \
    -v $XAUTH:$XAUTH \
    -e XAUTHORITY=$XAUTH \
    -v $CODE_HOST:$CODE_CONTAINER \
    --shm-size 8G \
    $IMAGE
xhost -local:docker

