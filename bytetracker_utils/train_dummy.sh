TRAIN_SCRIPT=../ByteTrack/tools/train.py
EXP_FILE=dummy_exp_nano.py
CHECKPOINT=../pretrained/yolox_nano.pth

python3 $TRAIN_SCRIPT -f $EXP_FILE -c $CHECKPOINT -d 1 -b 1 --fp16 -o 