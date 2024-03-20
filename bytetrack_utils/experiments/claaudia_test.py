import torch
import torch.distributed as dist
#import torch.nn as nn
import os
import json
#import sys

from yolox.exp import Exp as YoloxBaseExp
from yolox.data import (
    get_yolox_datadir,
    #MOTDataset,
    TrainTransform,
    ValTransform,
    YoloBatchSampler,
    DataLoader,
    InfiniteSampler,
    MosaicDetection,
)

#from yolox.evaluators import COCOEvaluator 
from eval.coco_evaluator import COCOEvaluator
from datasets.mot import MOTDataset

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

        # ---------------- dataloader config ---------------- #
        # set worker to 4 for shorter dataloader init time
        self.data_num_workers = 20
        self.input_size = (640, 1024)
        self.random_size = (10, 20)
        self.dataset_dir = os.path.join(get_yolox_datadir(), "traffic_dataset")
        self.train_json = "train.json"
        self.train_data_dir = "train"
        self.test_json = "test.json"
        self.test_data_dir = "test"
        #self.train_ann = "train.json"
        #self.val_ann = "test.json"

         # ---------------- model config ---------------- #
        self.depth = 1.33
        self.width = 1.25
        
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
        self.eval_interval = 3
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        self.output_dir = os.path.join(workspace, "YOLOX_outputs")
        
        # -----------------  testing config ------------------ #
        self.num_classes = self._get_num_of_classes() # TODO get from train JSONs
        self.test_size = (640, 1024) # test image size
        self.test_conf = 0.001 # confidence threshold (from 0 to 1, lower means more predictions)
        self.nmsthre = 0.65 # non-maximum supression threshold (from 0 to 1, higher means more predictions)
        self.track_thresh = 0.6 # tracking confidence threshold
        self.track_buffer = 30  # the frames for keep lost tracks
        self.match_thresh = 0.9 # matching threshold for tracking
        self.min_box_area = 100 # filter out tiny boxes
        
    
    def _get_num_of_classes(self):
        file_path = os.path.join(self.dataset_dir, "annotations", self.test_json)
        with open(file_path, "r") as json_file:
            data = json.load(json_file)
        num_of_classes = len(data['categories'])
        return num_of_classes

    def get_data_loader(self, batch_size, is_distributed, no_aug=False):
        dataset = MOTDataset(
            data_dir=self.dataset_dir,
            json_file=self.train_json,
            name=self.train_data_dir,
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


    #def get_eval_loader(self, batch_size, is_distributed, testdev=False):
    def get_eval_loader(self, batch_size, is_distributed):
        valdataset = MOTDataset(
            data_dir=self.dataset_dir,
            json_file=self.test_json,
            img_size=self.test_size,
            name=self.test_data_dir,
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


    #def get_evaluator(self, batch_size, is_distributed, testdev=False):
    def get_evaluator(self, batch_size, is_distributed, fp16):
        val_loader = self.get_eval_loader(batch_size, is_distributed)
        evaluator = COCOEvaluator(
            dataloader=val_loader,
            img_size=self.test_size,
            confthre=self.test_conf,
            nmsthre=self.nmsthre,
            num_classes=self.num_classes,
            output_dir=os.path.join(self.output_dir, self.exp_name),
            distributed=is_distributed,
            fp16=fp16
        )
    
        return evaluator
    
    #def eval(self, model, evaluator, is_distributed, half=False):
    def eval(self, evaluator, model):
        return evaluator.evaluate(model)
    

if __name__=="__main__":
    exp = Exp()
    exp.get_num_classes()