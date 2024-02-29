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

import os, shutil
import random
import cv2
import numpy as np


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        #self.num_classes = 1
        #self.depth = 0.67
        #self.width = 0.75
        #self.exp_name = "testing_dataloa"
        self.train_ann = "train.json"
        self.val_ann = "test.json"
        self.input_size = (640, 1024) #(640, 1024)
        self.test_size = (640, 1024)
        self.random_size = (18, 32)
        #self.max_epoch = 1
        #self.print_interval = 20
        #self.eval_interval = 1
        #self.test_conf = 0.001
        #self.nmsthre = 0.5
        #self.no_aug_epochs = 10
        #self.basic_lr_per_img = 0.0005
        #self.warmup_epochs = 0
        self.data_num_workers = 10
        self.output_dir = os.path.expanduser("~/YOLOX_outputs")
        os.environ["YOLOX_DATADIR"] = os.path.expanduser("~/datasets")


    def random_resize(self, data_loader, epoch, rank, is_distributed):
        tensor = torch.LongTensor(2).cuda()

        if rank == 0:
            size_factor = self.input_size[1] * 1.0 / self.input_size[0]
            size = random.randint(*self.random_size)
            size = (int(32 * size), 32 * int(size * size_factor))
            tensor[0] = size[0]
            tensor[1] = size[1]

        if is_distributed:
            dist.barrier()
            dist.broadcast(tensor, 0)

        input_size = data_loader.change_input_dim(
            multiple=(tensor[0].item(), tensor[1].item()), random_range=None
        )

        return input_size, tensor


    def get_data_loader(self, batch_size, is_distributed, use_mosaic_detection):
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
                preproc=TrainTransform()
            )

            if use_mosaic_detection:
                dataset = MosaicDetection(
                    dataset,
                    mosaic=True,
                    img_size=self.input_size,
                    preproc=TrainTransform(),
                    #preproc=TrainTransform(
                    #        rgb_means=(0.485, 0.456, 0.406),
                    #        std=(0.229, 0.224, 0.225),
                    #        max_labels=100,
                    #),
                    degrees=self.degrees,
                    translate=self.translate,
                    scale=self.scale,
                    shear=self.shear,
                    perspective=self.perspective,
                    enable_mixup=True,
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
                    mosaic=True,
                    #mosaic=not no_aug,
            )

            dataloader_kwargs = {"num_workers": self.data_num_workers, "pin_memory": True}
            dataloader_kwargs["batch_sampler"] = batch_sampler
            train_loader = DataLoader(self.dataset, **dataloader_kwargs)
            return train_loader, batch_sampler, sampler, dataset


    def get_eval_loader(self, batch_size, is_distributed, testdev=False):
            from yolox.data import MOTDataset, ValTransform

            val_dataset = MOTDataset(
                    data_dir=os.path.join(get_yolox_datadir(), "multi_view_mot"),
                    json_file=self.val_ann,
                    img_size=self.test_size,
                    name='test',
                    preproc=ValTransform(),
                    #preproc=ValTransform(
                    #    rgb_means=(0.485, 0.456, 0.406),
                    #    std=(0.229, 0.224, 0.225),
                    #),
            )

            if is_distributed:
                batch_size = batch_size // dist.get_world_size()
                sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
            else:
                sampler = torch.utils.data.SequentialSampler(val_dataset)

            dataloader_kwargs = {
                "num_workers": self.data_num_workers,
                "pin_memory": True,
                "sampler": sampler,
            }
            dataloader_kwargs["batch_size"] = batch_size
            val_loader = torch.utils.data.DataLoader(val_dataset, **dataloader_kwargs)
            return val_loader, sampler, val_dataset


def generate_random_color(seed):
    # Set seed for reproducibility
    random.seed(int(seed))
    # Generate random RGB values
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    # Return the random color as a tuple (R, G, B)
    return (r, g, b)


def draw_text(image, text, bbox):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 1
    # Calculate text size
    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
    text_width, text_height = text_size[0], text_size[1]
    # Calculate the position to put text (in the middle of the bounding box)
    text_x = bbox[0] + (bbox[2] - bbox[0]) // 2 - text_width // 2
    text_y = bbox[1] + (bbox[3] - bbox[1]) // 2 + text_height // 2
    # Put text on the image
    cv2.putText(image, text, (text_x, text_y), font, font_scale, (0, 0, 255), font_thickness)


def visualize_annotations(annotations, track_data, image_path, output):
    frame_id = track_data[2]
    video_id = track_data[3]
    file_name = track_data[4]
    metadata = f"FILE:{file_name} VIDEO_ID:{video_id} FRAME_ID:{frame_id}"
    print(f"Processing {metadata}")
    image = cv2.imread(image_path)
    image_metadata = np.zeros((50, image.shape[1], 3), dtype=np.uint8)
    cv2.putText(image_metadata, metadata, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    for ann in annotations:
        bbox = ann[0:4]  # Bounding box in format [x_min, y_min, width, height]
        bbox = list(map(int, [bbox[0], bbox[1], bbox[2], bbox[3]]))
        track_id = int(ann[5])
        cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), generate_random_color(track_id), 2)
        text = f"{track_id}"
        draw_text(image, text, bbox)

    # Concatenate image and metadata horizontally
    result_image = np.concatenate((image, image_metadata), axis=0)

    image_id = len(os.listdir(output)) + 1
    cv2.imwrite(os.path.join(output, str(image_id).zfill(6)+".png"), result_image)


if __name__=="__main__":
    DATASET = False
    SAMPLER = False
    BATCH_SAMPLER = False
    DATA_LOADER = False
    VALIDATION = False
    RESIZE = True
    
    exp = Exp()
    data_loader, batch_sampler, sampler, dataset = exp.get_data_loader(
        batch_size=1,
        is_distributed=False,
        use_mosaic_detection=False
    )
    val_loader, val_sampler, val_dataset = exp.get_eval_loader(
        batch_size=1,
        is_distributed=False,
    )

    if DATASET:
        dataset_dir = os.path.expanduser("~/datasets/multi_view_mot/train")
        output_dir = "./annotated_images_multi_view_from_index"
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)

        for i in range(0, len(dataset), 5):
            img, labels, img_info, _ = dataset[i]
            img = np.transpose(img, (1, 2, 0))[:, :, ::-1]
            #cv2.imshow("test", img)
            #cv2.waitKey()
            img = img*255
            image_id = len(os.listdir(output_dir)) + 1
            cv2.imwrite(os.path.join(output_dir, str(image_id).zfill(6)+".png"), img.astype(np.uint8))
            #visualize_annotations(res, img_info, os.path.join(dataset_dir, file_name), output_dir)
        #print(dataset.annotations)
            
    if SAMPLER:
        for sample in sampler:
            sample = sample.item()
            if sample > 1998:
                
                exit()

    if BATCH_SAMPLER:
        batch_id = 0
        for batch in batch_sampler:
            batch_id += 1
            print(batch)
            print(batch_id)
            if batch_id == 10:
                exit()

    if VALIDATION:
        for batch, val_batch in zip(data_loader, val_loader):
            image, labels, img_metadata, data_idx = batch 
            print(image.shape)
            print(labels.shape)
            print(img_metadata)
            print(data_idx)
            image, labels, img_metadata, data_idx = val_batch
            print(image.shape)
            print(labels.shape)
            print(img_metadata)
            print(data_idx) 
            exit()
    
    if RESIZE:
        iter = 1
        output = "./resized_images"
        os.makedirs(output, exist_ok=True)


        for batch in data_loader:
            batch_images = batch[0]
            image = batch_images[0].numpy()*255
            image = np.transpose(image, (1, 2, 0))
            image = image[:, :, ::-1] 
            iteration_data = f"iter {iter} image.shape {image.shape}"
            cv2.imwrite(f"{output}/{str(iter).zfill(5)}.png", image.astype(np.uint8))
            exp.random_resize(data_loader=data_loader, epoch = 20, rank=0, is_distributed=False)
            if iter % 50 == 0:
                exit()
            print(iteration_data)
            iter += 1
        #val_data_iter = iter(val_loader)
        #val_data = next(val_data_iter)  
        #print(val_data.shape)
