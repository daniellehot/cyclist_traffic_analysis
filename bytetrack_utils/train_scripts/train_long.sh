ROOT=~
TRAIN_SCRIPT=$ROOT/bytetrack_utils/train/train.py
EXP_FILE=$ROOT/bytetrack_utils/experiments/long_training_test.py
CHECKPOINT=$ROOT/pretrained/yolox_m.pth

python3 $TRAIN_SCRIPT -f $EXP_FILE -c $CHECKPOINT -d 1 -b 2 --fp16 -o