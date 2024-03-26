from loguru import logger

import torch
import torch.backends.cudnn as cudnn
#from torch.nn.parallel import DistributedDataParallel as DDP

from yolox.core import launch
from yolox.exp import get_exp
from yolox.utils import (
    #configure_nccl, 
    #fuse_model, 
    #get_local_rank, 
    get_model_info, 
    setup_logger
)
#from yolox.evaluators import MOTEvaluator
#from yolox.data import get_yolox_datadir
#from eval.coco_evaluator import COCOEvaluator
from inference.detector import Detector
from shared_utils.utils import default_launch, default_setup
from visualizer.visualizer import Visualizer

import argparse
import os
import random
import warnings
import cv2
#import glob
#import motmetrics as mm
#from collections import OrderedDict
#from pathlib import Path


def make_parser():
    parser = argparse.ArgumentParser("Inference")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")
    #parser.add_argument("-b", "--batch-size", type=int, default=1, help="batch size")
    parser.add_argument("-d", "--devices", default=1, type=int, help="device for training")
    parser.add_argument("-f", "--exp_file", default=None, type=str, help="pls input your expriment description file")
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
    parser.add_argument("--fp16", dest="fp16", default=False, action="store_true", help="Adopting mix precision evaluating.")
    parser.add_argument("--fuse", dest="fuse", default=False, action="store_true", help="Fuse conv and bn for testing.")
    parser.add_argument("opts", help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER)
    # distributed
    parser.add_argument("--dist-backend", default="nccl", type=str, help="distributed backend")
    parser.add_argument("--dist-url", default=None, type=str,help="url used to set up distributed training")
    parser.add_argument("--local_rank", default=0, type=int, help="local rank for dist training")
    parser.add_argument("--num_machines", default=1, type=int, help="num of node for training")
    parser.add_argument("--machine_rank", default=0, type=int, help="node rank for multi-node training")
    # det args
    parser.add_argument("--conf", default=None, type=float, help="test conf")
    parser.add_argument("--nms", default=None, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument("--seed", default=None, type=int, help="eval seed")
    # tracking args
    #parser.add_argument("--no_detection", action="store_true", help="only evaluate tracking")
    #parser.add_argument("--track_thresh", type=float, default=0.6, help="tracking confidence threshold")
    #parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    #parser.add_argument("--match_thresh", type=float, default=0.9, help="matching threshold for tracking")
    #parser.add_argument("--min-box-area", type=float, default=100, help='filter out tiny boxes')
    #parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")
    # input
    parser.add_argument("--video", default=None, type=str)
    parser.add_argument("--image", default=None, type=str)
    parser.add_argument("--folder", default=None, type=str)
    return parser

def process_video(video, detector, show=False, save=False, output="./"):
    print("TODO")

def process_folder(folder, detector, show=False, save=False, output="./"):
    image_extensions = ("png", "jpg", "jpeg")
    images = [f for f in os.listdir(folder) if f.endswith(image_extensions)]
    images = sorted(images) #sorted in case the folder is a MOT sequence
    print(images)
    for image in images:
        image_path = os.path.join(folder, image)
        process_image(image_path, detector, show, save)


def process_image(image, detector, show=False, save=False, output="./"):
    img = cv2.imread(image)
    detections = detector(img)
    img_detections = Visualizer.draw_model_output(detections, img)

    if show:
        cv2.imshow("detections", img_detections)
        cv2.waitKey()
    if save:
        print("TODO")




def main(exp, args, num_gpu):
    is_distributed, rank, file_name = default_setup(exp, args, num_gpu, None)

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)
    logger.info("Exp:\n{}".format(exp))

    detector = Detector(
        exp=exp,
        rank=rank,
        ckpt_file=os.path.join(file_name, "best_ckpt.pth.tar") if args.ckpt is None else args.ckpt,
        is_distributed=is_distributed,
        fuse=args.fuse,
        fp16=args.fp16
        )
    #model = detector.model
    #logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))

    if args.video is not None:
        process_video(args.video, detector)
    if args.folder is not None:
        process_folder(args.folder, detector, show=True)
    if args.image is not None:
        process_image(args.image, detector, show=True)


if __name__=="__main__":
    args = make_parser().parse_args()
    default_launch(args, main)