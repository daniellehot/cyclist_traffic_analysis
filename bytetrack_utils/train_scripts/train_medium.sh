#ROOT=~/cyclist_traffic_analysis
TRAIN_SCRIPT=~/bytetrack_utils/bytetrack_train.py
EXP_FILE=~/bytetrack_utils/experiments/multi_view_medium.py
CHECKPOINT=~/pretrained/bytetrack_m_mot17.pth.tar

python3 $TRAIN_SCRIPT -f $EXP_FILE -c $CHECKPOINT -d 1 -b 3 --fp16 -o