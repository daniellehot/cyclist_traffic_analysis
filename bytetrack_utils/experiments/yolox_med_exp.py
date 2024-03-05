import torch
import torch.distributed as dist
import torch.nn as nn

import os
import random

#from yolox.exp import BaseExp
from yolox.exp import Exp as YoloxBaseExp
from yolox.data import get_yolox_datadir

class Exp(YoloxBaseExp):
    def __init__(self):
        super().__init__()
        # ---------------- env config ---------------- #
        # handle container setup
        if os.path.exists(os.path.expanduser("~/cyclist_traffic_analysis")):
            workspace = os.path.expanduser("~/cyclist_traffic_analysis")
        else:
            workspace = os.path.expanduser("~")
        os.environ["YOLOX_DATADIR"] = os.path.join(workspace, "datasets")
        #os.environ["WORK_DIR"] = os.path.expanduser("~/cyclist_traffic_analysis")

        # ---------------- model config ---------------- #
        self.num_classes = 1
        self.depth = 0.67
        self.width = 0.75

        # ---------------- dataloader config ---------------- #
        # set worker to 4 for shorter dataloader init time
        self.data_num_workers = 10
        self.input_size = (640, 1024)
        self.random_size = (10, 20)
        self.train_ann = "train.json"
        self.val_ann = "test.json"
        
        # --------------- transform config ----------------- #
        self.degrees = 10.0
        self.translate = 0.1
        self.scale = (0.1, 2)
        self.mscale = (0.8, 1.6)
        self.shear = 2.0
        self.perspective = 0.0
        self.enable_mixup = True

        # --------------  training config --------------------- #
        self.seed = None
        self.warmup_epochs = 1
        self.max_epoch = 3
        self.warmup_lr = 0
        self.basic_lr_per_img = 0.01 / 64.0
        self.scheduler = "yoloxwarmcos"
        self.no_aug_epochs = 15
        self.min_lr_ratio = 0.05
        self.ema = True

        self.weight_decay = 5e-4
        self.momentum = 0.9
        self.print_interval = 10
        self.eval_interval = 1
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        self.output_dir = os.path.join(workspace, "YOLOX_outputs")
        
        # -----------------  testing config ------------------ #
        self.test_size = (640, 1024)
        self.test_conf = 0.001
        self.nmsthre = 0.65
        

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
            preproc=TrainTransform(),
        )

        dataset = MosaicDetection(
            dataset,
            mosaic=not no_aug,
            img_size=self.input_size,
            preproc=TrainTransform(),
            degrees=self.degrees,
            translate=self.translate,
            scale=self.scale,
            shear=self.shear,
            perspective=self.perspective,
            enable_mixup=self.enable_mixup,
        )
        
        self.dataset = dataset

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()

        sampler = InfiniteSampler(len(self.dataset), seed=self.seed if self.seed else 0)

        batch_sampler = YoloBatchSampler(
            sampler=sampler,
            batch_size=batch_size,
            drop_last=False,
            input_dimension=self.input_size,
            mosaic=not no_aug,
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
            name='test',
            preproc=ValTransform(),
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