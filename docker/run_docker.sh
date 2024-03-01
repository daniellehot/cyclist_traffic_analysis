ROOT=~/cyclist_traffic_analysis
PRETRAINED=$ROOT/pretrained
DATASETS=$ROOT/datasets
YOLOX_OUTPUTS=$ROOT/YOLOX_outputs
UTILS=$ROOT/bytetrack_utils
MULTI_VIEW=$ROOT/multi-view-data

#DOCKER_ROOT=/home/user/cyclist_traffic_analysis
DOCKER_ROOT=/home/user

#[ -z "$1" ] || CODE_CONTAINER=$1 # replace CODE_CONTAINER in case there is any 
IMAGE=dale97/bytetrack

xhost +local:docker
XSOCK=/tmp/.X11-unix
XAUTH=/tmp/.docker.xauth
docker run --gpus all -it --rm \
    -v $PRETRAINED:$DOCKER_ROOT/pretrained \
    -v $DATASETS:$DOCKER_ROOT/datasets \
    -v $YOLOX_OUTPUTS:$DOCKER_ROOT/YOLOX_outputs \
    -v $UTILS:$DOCKER_ROOT/bytetrack_utils \
    -v $MULTI_VIEW:$DOCKER_ROOT/multi-view-data \
    -e DISPLAY=$DISPLAY \
    -e XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR \
    -e XAUTHORITY=$XAUTH \
    -v $XSOCK:$XSOCK \
    -v $XAUTH:$XAUTH \
    --device /dev/video0:/dev/video0:mwr \
    --net=host \
    --shm-size 8G \
    --privileged \
    -w $DOCKER_ROOT \
    $IMAGE
xhost -local:docker