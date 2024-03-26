from loguru import logger
import torch
#import torch.backends.cudnn as cudnn
from torch.nn.parallel import DistributedDataParallel as DDP
import torchvision.transforms as transforms
#import argparse
#import os
#import random
#import warnings

#from yolox.core import launch
#from yolox.exp import get_exp
from yolox.utils import (
    #configure_nccl, 
    fuse_model, 
    #get_local_rank, 
    #get_model_info, 
    #setup_logger
)
#from eval.coco_evaluator import COCOEvaluator
from shared_utils.utils import postprocess


class Detector:
    def __init__(self, exp, rank, ckpt_file, is_distributed, fuse, fp16):
        self.exp = exp
        self.model = exp.get_model()

        torch.cuda.set_device(rank)
        self.model.cuda(rank)

        logger.info(f"Loading checkpoint {ckpt_file} ...")
        loc = "cuda:{}".format(rank)
        ckpt = torch.load(ckpt_file, map_location=loc)
        # load the model state dict
        self.model.load_state_dict(ckpt["model"])
        logger.info("Checkpoint loaded")

        if is_distributed:
            logger.info("\tDistributed model...")
            self.model = DDP(self.model, device_ids=[rank])

        if fuse:
            logger.info("\tFusing model...")
            self.model = fuse_model(self.model)
        
        """
        if fp16:
            logger.info("\tHalf model...")
            self.model.half()
            self.tensor_type = torch.cuda.HalfTensor
        else:
            self.tensor_type = torch.cuda.FloatTensor
        """
        self.fp16 = fp16
        self.model.eval()   

    
    def __call__(self, img, img_size = None):
        if self.fp16:
            logger.info("model.half()")
            self.model.half()
            self.tensor_type = torch.cuda.HalfTensor
        else:
            self.tensor_type = torch.cuda.FloatTensor

        transformations = [transforms.ToPILImage()]
        
        if img_size is not None:
            transformations.append(transforms.Resize((img_size[0], img_size[1])))
        else:
            transformations.append(transforms.Resize(self.exp.test_size))
            
        transformations.append(transforms.ToTensor())
        transform = transforms.Compose(transformations)
    
        img_tensor = transform(img).cuda().unsqueeze(0) 
        img_tensor = img_tensor.type(self.tensor_type)
            
        with torch.no_grad():
            output = self.model(img_tensor)
            output = postprocess(output, self.exp.num_classes, self.exp.test_conf, self.exp.nmsthre)[0].cpu().numpy()
        return output