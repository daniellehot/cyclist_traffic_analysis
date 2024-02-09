import os, sys
sys.path.append("/repos/DINO")
import torch, json
import numpy as np
from PIL import Image

from main import build_model_main
from util.slconfig import SLConfig
from datasets import build_dataset
import datasets.transforms as T
from util.visualizer import COCOVisualizer
from util import box_ops

model_config_path = "/repos/DINO/config/DINO/DINO_5scale.py" # change the path of the model config file
model_checkpoint_path = "/home/daniel/cyclist_traffic_analysis/weights/dino_r50_scale5.pth" # change the path of the model checkpoint
# See our Model Zoo section in README.md for more details about our pretrained models.

args = SLConfig.fromfile(model_config_path) 
args.device = 'cuda' 
model, criterion, postprocessors = build_model_main(args)
checkpoint = torch.load(model_checkpoint_path, map_location='cpu')
model.load_state_dict(checkpoint['model'])
_ = model.eval()

# load coco names
with open('/repos/DINO/util/coco_id2name.json') as f:
    id2name = json.load(f)
    id2name = {int(k):v for k,v in id2name.items()}

# load image
image = Image.open("office.jpg").convert("RGB") # load image

# transform images
transform = T.Compose([
    #T.RandomResize([800], max_size=1333),
    T.RandomResize([400]),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
image, _ = transform(image, None)

# predict images
output = model.cuda()(image[None].cuda())
output = postprocessors['bbox'](output, torch.Tensor([[1.0, 1.0]]).cuda())[0]
print(output)
# visualize outputs
thershold = 0.3 # set a thershold

#vslzr = COCOVisualizer()

#scores = output['scores']
#labels = output['labels']
#boxes = box_ops.box_xyxy_to_cxcywh(output['boxes'])
#select_mask = scores > thershold

#box_label = [id2name[int(item)] for item in labels[select_mask]]
#pred_dict = {
#    'boxes': boxes[select_mask],
#    'size': torch.Tensor([image.shape[1], image.shape[2]]),
#    'box_label': box_label
#}
#vslzr.visualize(image, pred_dict, savedir=None, dpi=100)