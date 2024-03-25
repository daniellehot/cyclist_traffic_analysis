from loguru import logger

import torch
import torch.backends.cudnn as cudnn
from torch.nn.parallel import DistributedDataParallel as DDP

from yolox.core import launch
from yolox.exp import get_exp
from yolox.utils import (
    configure_nccl, 
    fuse_model, 
    get_local_rank, 
    get_model_info, 
    setup_logger
)
#from yolox.evaluators import MOTEvaluator
#from yolox.data import get_yolox_datadir

import argparse
import os
import random
import warnings
import glob
import motmetrics as mm
from collections import OrderedDict
from pathlib import Path

from eval.mot_evaluator import MOTEvaluator
from inference.detector import Detector


def make_parser():
    parser = argparse.ArgumentParser("YOLOX Tracking Eval")
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
    parser.add_argument("--cache", action="store_true", help="only evaluate tracking")
    parser.add_argument("--track_thresh", type=float, default=None, help="tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=None, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=None, help="matching threshold for tracking")
    parser.add_argument("--min-box-area", type=float, default=None, help='filter out tiny boxes')
    #parser.add_argument("--track_thresh", type=float, default=0.6, help="tracking confidence threshold")
    #parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    #parser.add_argument("--match_thresh", type=float, default=0.9, help="matching threshold for tracking")
    #parser.add_argument("--min-box-area", type=float, default=100, help='filter out tiny boxes')
    #parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")
    return parser


@logger.catch
def main(exp, args, num_gpu):
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn(
            "You have chosen to seed testing. This will turn on the CUDNN deterministic setting, "
        )

    is_distributed = num_gpu > 1

    # set environment variables for distributed training
    cudnn.benchmark = True

    rank = args.local_rank
    # rank = get_local_rank()

    file_name = os.path.join(exp.output_dir, args.experiment_name)

    if rank == 0:
        os.makedirs(file_name, exist_ok=True)

    #results_folder = os.path.join(file_name, "track_results")
    #os.makedirs(results_folder, exist_ok=True)

    setup_logger(file_name, distributed_rank=rank, filename="val_log.txt", mode="a")
    logger.info("Args: {}".format(args))

    # overwrite thresholds from the experiment file with arguments if arguments exist
    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)
    if args.track_thresh is not None:
        exp.track_thresh = args.track_thresh
    if args.track_buffer is not None:
        exp.track_buffer = args.track_buffer
    if args.match_thresh is not None:
        exp.match_thresh = args.match_thresh
    if args.min_box_area is not None:
        exp.min_box_area = args.min_box_area

    evaluator = MOTEvaluator(
        dataloader=exp.get_eval_loader(args.batch_size, is_distributed),
        img_size=exp.test_size, 
        confthre=exp.test_conf, 
        nmsthre=exp.nmsthre,
        num_classes=exp.num_classes,
        output_dir=file_name,
        track_thresh=exp.track_thresh, 
        track_buffer=exp.track_buffer, 
        match_thresh=exp.match_thresh, 
        min_box_area=exp.min_box_area,
        distributed=is_distributed, 
        fp16=args.fp16,
    )
    logger.info("evaluator :\n{}".format(evaluator))

    if not args.cache:    
        detector = Detector(
            exp=exp,
            rank=rank,
            ckpt_file=os.path.join(file_name, "best_ckpt.pth.tar") if args.ckpt is None else args.ckpt,
            is_distributed=is_distributed,
            fuse=args.fuse,
            fp16=args.fp16
            )
        model = detector.model
    
        logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))
        #logger.info("Model Structure:\n{}".format(str(model)))
    
        ap50_95, ap50, summary = evaluator.evaluate(model)
        logger.info("Detection summary \n" + summary)
    
    logger.info("Evaluating Tracking Performance")
    summary_tracking = evaluator.evaluate_tracking()
    logger.info("Tracking summary \n" + summary_tracking)
    logger.info('Completed')

if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)
    exp.merge(args.opts)

    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    num_gpu = torch.cuda.device_count() if args.devices is None else args.devices
    assert num_gpu <= torch.cuda.device_count()

    launch(
        main,
        num_gpu,
        args.num_machines,
        args.machine_rank,
        backend=args.dist_backend,
        dist_url=args.dist_url,
        args=(exp, args, num_gpu),
    )
