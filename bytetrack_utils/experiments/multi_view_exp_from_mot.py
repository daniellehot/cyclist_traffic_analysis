# encoding: utf-8
import os, shutil
import random
import torch
import torch.nn as nn
import torch.distributed as dist

from yolox.exp import Exp as MyExp
from yolox.data import get_yolox_datadir


def show_some_training_images(train_loader, images_to_save=100, output="./training_images"):
    import cv2
    import numpy as np
    #import torchvision.transforms.functional as TF

    if os.path.exists(output):
        shutil.rmtree(output)
    os.makedirs(output, exist_ok=True)

    TAB = "   "
    saved = 0
    for batch in train_loader:
        # Convert images from tensor to numpy array
        images = batch[0]
        labels = batch[1]   
        print(batch[2])
        print(batch[3])
        exit()
        for i in range(images.shape[0]):
            image = images[i]
            image = image.numpy()*255
            image = np.transpose(image, (1, 2, 0))
            image = image[:, :, ::-1] #RGB to BGR

            image_labels = labels[i]
            saved += 1
            cv2.imwrite(f"{output}/{str(saved).zfill(5)}.png", image.astype(np.uint8))
            print(TAB, f"Saving image {saved}")
        if saved > images_to_save:
            break
    exit()        

class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.num_classes = 1
        #self.depth = 1.33
        #self.width = 1.25
        self.depth = 0.67
        self.width = 0.75
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        self.train_ann = "train.json"
        self.val_ann = "val.json"
        self.input_size = (320, 512) #(640, 1024)
        self.test_size = (320, 512)
        self.random_size = (18, 32)
        self.max_epoch = 15
        self.print_interval = 20
        self.eval_interval = 5
        self.test_conf = 0.1
        self.nmsthre = 0.7
        self.no_aug_epochs = 10
        self.basic_lr_per_img = 0.001 / 64.0
        self.warmup_epochs = 1

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
        print("MOTDataset", len(dataset))
        print(dataset, "\n")
        
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

        sampler = InfiniteSampler(
            len(self.dataset), seed=self.seed if self.seed else 0
        )

        batch_sampler = YoloBatchSampler(
            sampler=sampler,
            batch_size=batch_size,
            drop_last=False,
            input_dimension=self.input_size,
            mosaic=True,
            #mosaic=not no_aug,
        )

        dataloader_kwargs = {"num_workers": self.data_num_workers, "pin_memory": True}
        dataloader_kwargs["batch_sampler"] = batch_sampler
        train_loader = DataLoader(self.dataset, **dataloader_kwargs)
        show_some_training_images(train_loader)
        
        return train_loader

    def get_eval_loader(self, batch_size, is_distributed, testdev=False):
        from yolox.data import MOTDataset, ValTransform

        valdataset = MOTDataset(
            data_dir=os.path.join(get_yolox_datadir(), "multi_view_mot"),
            json_file=self.val_ann,
            img_size=self.test_size,
            name='train',
            preproc=ValTransform(
                rgb_means=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        )

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()
            sampler = torch.utils.data.distributed.DistributedSampler(
                valdataset, shuffle=False
            )
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