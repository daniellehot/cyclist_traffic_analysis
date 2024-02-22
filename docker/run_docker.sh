USER=daniel
CODE_PRETRAINED=/home/daniel/cyclist_traffic_analysis/pretrained
CODE_DATASETS=/home/daniel/cyclist_traffic_analysis/datasets
CODE_YOLOX_OUTPUTS=/home/daniel/cyclist_traffic_analysis/YOLOX_outputs
CODE_UTILS=/home/daniel/cyclist_traffic_analysis/bytetracker_utils
#[ -z "$1" ] || CODE_CONTAINER=$1 # replace CODE_CONTAINER in case there is any 
IMAGE=bytetrack

xhost +local:docker
XSOCK=/tmp/.X11-unix
XAUTH=/tmp/.docker.xauth
docker run --gpus all -it --rm \
    -v $CODE_PRETRAINED:/workspace/ByteTrack/pretrained \
    -v $CODE_DATASETS:/workspace/ByteTrack/datasets \
    -v $CODE_YOLOX_OUTPUTS:/workspace/ByteTrack/YOLOX_outputs \
    -v $CODE_UTILS:/workspace/ByteTrack/bytetracker_utils \
    -e DISPLAY=$DISPLAY \
    -e XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR \
    -e XAUTHORITY=$XAUTH \
    -v $XSOCK:$XSOCK \
    -v $XAUTH:$XAUTH \
    --device /dev/video0:/dev/video0:mwr \
    --net=host \
    --shm-size 8G \
    --privileged \
    $IMAGE
xhost -local:docker

#echo "Reclaiming all files created by the container with sudo chown -Rc $USER $CODE_HOST"
#sudo chown -Rc $USER $CODE_HOST