#!/bin/bash
BYTETRACK_UTILS_PATH=~/cyclist_traffic_analysis/bytetrack_utils
BYTETRACK_UTILS_DOCKER_PATH=~/bytetrack_utils
PATH_TO_ADD=$BYTETRACK_UTILS_PATH

# Check if the provided path exists
if [ ! -d "$BYTETRACK_UTILS_PATH" ]; then
    echo "INFO: $BYTETRACK_UTILS_PATH does not exist. Using Docker path $BYTETRACK_UTILS_DOCKER_PATH"
    PATH_TO_ADD=$BYTETRACK_UTILS_DOCKER_PATH
fi

# Add the provided path to PYTHONPATH
export PYTHONPATH="$PYTHONPATH:$PATH_TO_ADD"

echo "Added $PATH_TO_ADD to PYTHONPATH"
