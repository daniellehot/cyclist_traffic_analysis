apt install wget
cd detrex
# download pretrained DINO model
wget https://github.com/IDEA-Research/detrex-storage/releases/download/v0.2.1/dino_r50_4scale_12ep.pth
# download the demo image
wget https://github.com/IDEA-Research/detrex-storage/releases/download/v0.2.1/idea.jpg
# run demo
CONFIG=projects/dino/configs/dino-resnet/dino_r50_4scale_12ep.py
INPUT=./idea.jpg
OUTPUT=./demo_output.jpg
OPTS=train.init_checkpoint=./dino_r50_4scale_12ep.pth
python3 demo/demo.py --config-file $CONFIG --input $INPUT --output $OUTPUT --opts $OPTS 
mv dexter/demo_output.jpg /home/daniel/cyclist_traffic_analysis/