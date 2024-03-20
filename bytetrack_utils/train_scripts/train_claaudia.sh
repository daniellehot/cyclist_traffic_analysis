ROOT=~/cyclist_traffic_analysis
TRAIN_SCRIPT=$ROOT/bytetrack_utils/train/train.py
EXP_FILE=$ROOT/bytetrack_utils/experiments/claaudia_test.py
CHECKPOINT=$ROOT/pretrained/yolox_x.pth

python3 $TRAIN_SCRIPT -f $EXP_FILE -c $CHECKPOINT -d 1 -b 8 --fp16 -o