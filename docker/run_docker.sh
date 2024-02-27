ROOT=~/cyclist_traffic_analysis
#DOCKER_ROOT=/home/user/cyclist_traffic_analysis
DOCKER_ROOT=/home/user
CODE_PRETRAINED=$ROOT/pretrained
CODE_DATASETS=$ROOT/datasets
CODE_YOLOX_OUTPUTS=$ROOT/YOLOX_outputs
CODE_UTILS=$ROOT/bytetrack_utils
#[ -z "$1" ] || CODE_CONTAINER=$1 # replace CODE_CONTAINER in case there is any 
IMAGE=dale97/bytetrack

xhost +local:docker
XSOCK=/tmp/.X11-unix
XAUTH=/tmp/.docker.xauth
docker run --gpus all -it --rm \
    -v $CODE_PRETRAINED:$DOCKER_ROOT/pretrained \
    -v $CODE_DATASETS:$DOCKER_ROOT/datasets \
    -v $CODE_YOLOX_OUTPUTS:$DOCKER_ROOT/YOLOX_outputs \
    -v $CODE_UTILS:$DOCKER_ROOT/bytetrack_utils \
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