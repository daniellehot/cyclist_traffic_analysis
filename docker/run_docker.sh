USER=daniel
CODE_HOST=/home/daniel/cyclist_traffic_analysis/
CODE_CONTAINER=/home/daniel/cyclist_traffic_analysis/
[ -z "$1" ] || CODE_CONTAINER=$1 # replace CODE_CONTAINER in case there is any 
#DATA_HOST=/home/daniel/OneDrive/autofish_groups/
#DATA_CONTAINER=/home/create.aau.dk/vo65hs/autofish_groups/
IMAGE=dale97/detrex_container

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

echo "Reclaiming all files created by the container with sudo chown -Rc $USER $CODE_HOST"
sudo chown -Rc $USER $CODE_HOST
