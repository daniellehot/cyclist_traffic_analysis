from loguru import logger
import torch
import torch.backends.cudnn as cudnn
from torch.nn.parallel import DistributedDataParallel as DDP
import argparse
import os
import random
import warnings

from yolox.core import launch
from yolox.exp import get_exp
from yolox.utils import (
    #configure_nccl, 
    fuse_model, 
    #get_local_rank, 
    get_model_info, 
    setup_logger
)
#from eval.coco_evaluator import COCOEvaluator
from shared_utils.utils import postprocess




class Detector:
    def __init__(self, args):
        # create an experiment object
        self.exp = get_exp(args.exp_file, args.name)
        self.exp.merge(args.opts)

        # GPU available? 
        num_gpu = torch.cuda.device_count() if args.devices is None else args.devices
        if num_gpu > 1:
            print("Detector is not supported for multiple GPUs")
            exit()
        if num_gpu == 0:
            print("CPU-only detector is not supported")
            exit()
        
        # overwrite experiment thresholds with input arguments 
        if args.conf is not None:
            self.exp.test_conf = args.conf
        if args.nms is not None:
            self.exp.nmsthre = args.nms
        if args.tsize is not None:
            self.exp.test_size = (args.tsize, args.tsize)

        # create GPU-connected model
        self.model = self.exp.get_model()
        torch.cuda.set_device(args.local_rank)
        self.model.cuda(args.local_rank)
        self.model.eval()
        
        # load in a checkpoint
        if not args.experiment_name:
            args.experiment_name = self.exp.exp_name
        file_name = os.path.join(self.exp.output_dir, args.experiment_name)

        if args.ckpt is None:
            ckpt_file = os.path.join(file_name, "best_ckpt.pth.tar")
        else:
            ckpt_file = args.ckpt
    
        logger.info("loading checkpoint")
        loc = "cuda:{}".format(args.local_rank)
        ckpt = torch.load(ckpt_file, map_location=loc)
        self.model.load_state_dict(ckpt["model"])
        logger.info("loaded checkpoint done.")

        # fuse arguments (whatever this does)
        if args.fuse:
            logger.info("\tFusing model...")
            self.model = fuse_model(self.model)
    
    
    def predict(self, image):
        print("TODO")


def make_parser():
    parser = argparse.ArgumentParser("YOLOX Eval")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")
    parser.add_argument("-b", "--batch-size", type=int, default=1, help="batch size")
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
    #parser.add_argument("--trt", dest="trt", default=False, action="store_true", help="Using TensorRT model for testing.")
    #parser.add_argument("--test", dest="test", default=False, action="store_true", help="Evaluating on test-dev set.")
    #parser.add_argument("--speed", dest="speed", default=False, action="store_true", help="speed test only.")
    # det args
    #parser.add_argument("--conf", default=0.01, type=float, help="test conf")
    #parser.add_argument("--nms", default=0.7, type=float, help="test nms threshold")
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
    return parser



if __name__ == "__main__":
    args = make_parser().parse_args()
    detector = Detector(args)

"""


"""