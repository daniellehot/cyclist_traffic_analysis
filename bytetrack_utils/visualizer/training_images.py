from loguru import logger
import torch
import torch.backends.cudnn as cudnn
from yolox.core import launch
from yolox.exp import get_exp
import argparse
import random
import warnings
import numpy as np
import cv2
import os

# Custom
from train.yolox_trainer_modified import TrainerModified

def make_parser():
    parser = argparse.ArgumentParser("YOLOX train parser")
    parser.add_argument("-f", "--exp_file", default=None, type=str, help="plz input your expriment description file")
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")
    parser.add_argument("opts", help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER)
    #parser.add_argument("-b", "--batch-size", type=int, default=64, help="batch size")
    #parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    """
    parser.add_argument("--fp16", dest="fp16", default=True, action="store_true", help="Adopting mix precision training.")
    parser.add_argument("--local_rank", default=0, type=int, help="local rank for dist training")
    parser.add_argument("--dist-backend", default="nccl", type=str, help="distributed backend")
    parser.add_argument("--dist-url", default=None, type=str, help="url used to set up distributed training")
    parser.add_argument("-d", "--devices", default=None, type=int, help="device for training")
    parser.add_argument("--resume", default=False, action="store_true", help="resume training")
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="checkpoint file")
    parser.add_argument("-e", "--start_epoch", default=None, type=int, help="resume training start epoch")
    parser.add_argument("--num_machines", default=1, type=int, help="num of node for training")
    parser.add_argument("--machine_rank", default=0, type=int, help="node rank for multi-node training")
    parser.add_argument("-o", "--occupy", dest="occupy", default=False, action="store_true", help="occupy GPU memory first for training.")
    """
    return parser


def save_dataloader_images(dataloader_iter, file_name, output="./dataloader_images"):
    os.makedirs(output, exist_ok=True)
    rand_idx = random.randint(0, 100)
    for i in range(rand_idx):
        print(f"Iterating over dataloader  {i}/{rand_idx}")
        next(dataloader_iter)
    batch = next(dataloader_iter)
    image = batch[0][0].numpy()*255
    image = np.transpose(image, (1, 2, 0))
    image = image[:, :, ::-1] 
    cv2.imwrite(os.path.join(output, file_name), image.astype(np.uint8))
    print(f"Saving image {file_name}. Image shape {image.shape}")


def main(args):
    distributed = False
    rank = 0
    batch_size = 1
    
    exp = get_exp(args.exp_file, args.name)
    exp.merge(args.opts)
    dataloader = exp.get_data_loader(batch_size, is_distributed=distributed, no_aug=False)
    dataloader_iter = iter(dataloader)

    sample_size = 20
    to_save = sorted([random.randint(0, len(dataloader)) for i in range(sample_size)])
    print(f"To save {to_save}")
    resize_iteration = 250

    for idx, batch in enumerate(dataloader):
        resizing = False
        saving_image = False
        if idx % resize_iteration == 0:
            exp.random_resize(dataloader, -1, rank, distributed)
            dataloader_iter = iter(dataloader)
            resizing = True
        if idx in to_save:
            save_dataloader_images(dataloader_iter, file_name =f"{idx}.png")
            saving_image = True
        print(f"Index {idx} Resizing {resizing} Saving Image {saving_image}")


    #max_itr = 15
    #resize_itr = 1
    #for current_itr in range(1, max_itr+1):
    #    if current_itr % resize_itr == 0:
    #        exp.random_resize(dataloader, -1, rank, distributed)
    #        dataloader_iter = iter(dataloader)
    #    save_dataloader_images(dataloader_iter, file_name =f"{current_itr}.png")


if __name__=="__main__":
    args = make_parser().parse_args()
    print(args)
    main(args)