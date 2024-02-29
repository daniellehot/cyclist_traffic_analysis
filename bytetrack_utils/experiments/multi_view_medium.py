# encoding: utf-8
import os, shutil
import random
import torch
import torch.nn as nn
import torch.distributed as dist
from yolox.exp import Exp as MyExp
from yolox.data import get_yolox_datadir
import os

import torch
import torch.distributed as dist
import torch.nn as nn

import os
import random

class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.num_classes = 1
        self.depth = 0.67
        self.width = 0.75
        self.exp_name = "yolox_medium"
        self.train_ann = "train.json"
        self.val_ann = "test.json"
        self.input_size = (320, 512) #(640, 1024)
        self.test_size = (320, 512)
        self.random_size = (18, 32)
        self.max_epoch = 1
        self.print_interval = 20
        self.eval_interval = 1
        self.test_conf = 0.001
        self.nmsthre = 0.5
        self.no_aug_epochs = 10
        self.basic_lr_per_img = 0.0005
        self.warmup_epochs = 0
        self.data_num_workers = 10
        self.output_dir = os.path.expanduser("~/YOLOX_outputs")
        os.environ["YOLOX_DATADIR"] = os.path.expanduser("~/datasets")


    def get_data_loader(self, batch_size, is_distributed, no_aug=False):
            from yolox.data import (
                    MOTDataset,
                    TrainTransform,
                    YoloBatchSampler,
                    DataLoader,
                    InfiniteSampler,
                    MosaicDetection,
            )

            dataset = MOTDataset(
                    data_dir=os.path.join(get_yolox_datadir(), "multi_view_mot"),
                    json_file=self.train_ann,
                    name='train',
                    img_size=self.input_size,
                    preproc=TrainTransform(
                        #rgb_means=(0.485, 0.456, 0.406),
                        #std=(0.229, 0.224, 0.225),
                        max_labels=40,
                    ),
            )

            """
            dataset = MosaicDetection(
                dataset,
                mosaic=not no_aug,
                img_size=self.input_size,
                preproc=TrainTransform(
                        #rgb_means=(0.485, 0.456, 0.406),
                        #std=(0.229, 0.224, 0.225),
                        max_labels=100,
                ),
                degrees=self.degrees,
                translate=self.translate,
                scale=self.scale,
                shear=self.shear,
                perspective=self.perspective,
                enable_mixup=self.enable_mixup,
                )
                print("MosaicDetection", len(dataset))
                print(dataset, "\n")
            """

            self.dataset = dataset

            if is_distributed:
                    batch_size = batch_size // dist.get_world_size()

            sampler = InfiniteSampler(len(self.dataset), seed=self.seed if self.seed else 0)

            batch_sampler = YoloBatchSampler(
                    sampler=sampler,
                    batch_size=batch_size,
                    drop_last=False,
                    input_dimension=self.input_size,
                    mosaic=False,
                    #mosaic=not no_aug,
            )

            dataloader_kwargs = {"num_workers": self.data_num_workers, "pin_memory": True}
            dataloader_kwargs["batch_sampler"] = batch_sampler
            train_loader = DataLoader(self.dataset, **dataloader_kwargs)
            return train_loader


    def get_eval_loader(self, batch_size, is_distributed, testdev=False):
            from yolox.data import MOTDataset, ValTransform

            valdataset = MOTDataset(
                    data_dir=os.path.join(get_yolox_datadir(), "multi_view_mot"),
                    json_file=self.val_ann,
                    img_size=self.test_size,
                    name='train',
                    #preproc=None,
                    preproc=ValTransform(
                        rgb_means=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225),
                    ),
            )

            if is_distributed:
                    batch_size = batch_size // dist.get_world_size()
                    sampler = torch.utils.data.distributed.DistributedSampler(valdataset, shuffle=False)
            else:
                    sampler = torch.utils.data.SequentialSampler(valdataset)

            dataloader_kwargs = {
                    "num_workers": self.data_num_workers,
                    "pin_memory": True,
                    "sampler": sampler,
            }
            dataloader_kwargs["batch_size"] = batch_size
            val_loader = torch.utils.data.DataLoader(valdataset, **dataloader_kwargs)
            return val_loader


    def get_evaluator(self, batch_size, is_distributed, testdev=False):
            from yolox.evaluators import COCOEvaluator

            val_loader = self.get_eval_loader(batch_size, is_distributed, testdev=testdev)
            evaluator = COCOEvaluator(
                    dataloader=val_loader,
                    img_size=self.test_size,
                    confthre=self.test_conf,
                    nmsthre=self.nmsthre,
                    num_classes=self.num_classes,
                    testdev=testdev,
            )
            return evaluator