#ROOT=~/cyclist_traffic_analysis
TRAIN_SCRIPT=~/bytetrack_utils/bytetrack_train.py
EXP_FILE=~/bytetrack_utils/experiments/multi_view_exp_nano.py
CHECKPOINT=~/pretrained/yolox_nano.pth

python3 $TRAIN_SCRIPT -f $EXP_FILE -c $CHECKPOINT -d 1 -b 3 --fp16 -o