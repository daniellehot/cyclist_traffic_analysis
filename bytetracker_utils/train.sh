#######################
# python3 tools/train.py -f exps/example/mot/yolox_x_mix_mot20_ch.py -d 8 -b 48 --fp16 -o -c pretrained/yolox_x.pth
#   -f STR path to an experiment description file
#   -d INT number of training devices (GPUs)
#   -b INT batch size
#   --fp16 BOOL "Adopting mix precision training."
#   -o BOOL "occupy GPU memory first for training."
#   -c STR path to a checkpoint
#######################


python3 train_modified.py -f multi_view_exp_from_mot.py -c $1 -d 1 -b 2 --fp16 -o