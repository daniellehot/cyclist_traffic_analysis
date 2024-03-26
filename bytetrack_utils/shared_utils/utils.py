import torch
import torchvision
import torch.backends.cudnn as cudnn
import random
import os
import warnings
from loguru import logger

from yolox.core import launch
from yolox.exp import get_exp
from yolox.utils import setup_logger


#def postprocess(prediction, num_classes, conf_thre=0.7, nms_thre=0.45):
def postprocess(prediction, num_classes, conf_thre, nms_thre):
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    output = [None for _ in range(len(prediction))]
    for i, image_pred in enumerate(prediction):

        # If none are remaining => process next image
        if not image_pred.size(0):
            continue
        # Get score and class with highest confidence
        class_conf, class_pred = torch.max(
            image_pred[:, 5 : 5 + num_classes], 1, keepdim=True
        )

        conf_mask = (image_pred[:, 4] * class_conf.squeeze() >= conf_thre).squeeze()
        # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
        detections = torch.cat((image_pred[:, :5], class_conf, class_pred.float()), 1)
        detections = detections[conf_mask]
        if not detections.size(0):
            continue

        nms_out_index = torchvision.ops.batched_nms(
            detections[:, :4],
            detections[:, 4] * detections[:, 5],
            detections[:, 6],
            nms_thre,
        )
        detections = detections[nms_out_index]
        if output[i] is None:
            output[i] = detections
        else:
            output[i] = torch.cat((output[i], detections))

    return output


def default_launch(args, to_launch):
    exp = get_exp(args.exp_file, args.name)
    exp.merge(args.opts)

    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    num_gpu = torch.cuda.device_count() if args.devices is None else args.devices
    assert num_gpu <= torch.cuda.device_count()

    launch(
        to_launch,
        num_gpu,
        args.num_machines,
        args.machine_rank,
        backend=args.dist_backend,
        dist_url=args.dist_url,
        args=(exp, args, num_gpu),
    )
    

def default_setup(exp, args, num_gpu, log_filename):
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn("You have chosen to seed testing. This will turn on the CUDNN deterministic setting, ")

    is_distributed = num_gpu > 1
    # set environment variables for distributed training
    cudnn.benchmark = True

    file_name = os.path.join(exp.output_dir, args.experiment_name)
    rank = args.local_rank
    if rank == 0:
        os.makedirs(file_name, exist_ok=True)

    if log_filename is not None:
        setup_logger(file_name, distributed_rank=rank, filename=log_filename, mode="a")
        #setup_logger(file_name, distributed_rank=rank, filename="val_log.txt", mode="a")
    logger.info("Args: {}".format(args))
    return is_distributed, rank, file_name



if __name__=="__main__":
    def test_function(exp, args, num_gpu):
        print("Inside test_function")
        print("exp:", exp)
        print("args:", args)
        print("num_gpu:", num_gpu)
        is_distributed, rank, file_name = default_setup(exp, args, num_gpu, None)
        print("is_distributed:", is_distributed)
        print("rank:", rank)
        print("file_name:", file_name)

    import argparse
    parser = argparse.ArgumentParser("shared_utils main")
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
    # det args
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
    default_launch(args=parser.parse_args(), to_launch=test_function)
