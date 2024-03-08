#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

from loguru import logger
from tqdm import tqdm

import torch

from yolox.utils import (
    gather,
    is_main_process,
    #postprocess,
    synchronize,
    time_synchronized,
    xyxy2xywh
)

import contextlib
import io
import itertools
import json
#import tempfile
import time
import os, shutil
import csv

from shared_utils.utils import postprocess


class COCOEvaluator:
    """
    COCO AP Evaluation class.  All the data in the val2017 dataset are processed
    and evaluated by COCO API.
    """

    def __init__(self, dataloader, img_size, confthre, nmsthre, num_classes, output_dir, distributed=False, fp16=False):
        """
        Args:
            dataloader (Dataloader): evaluate dataloader.
            img_size (int): image size after preprocess. images are resized
                to squares whose shape is (img_size, img_size).
            confthre (float): confidence threshold ranging from 0 to 1, which
                is defined in the config file.
            nmsthre (float): IoU threshold of non-max supression ranging from 0 to 1.
        """
        self.dataloader = dataloader
        self.img_size = img_size
        self.confthre = confthre
        self.nmsthre = nmsthre
        self.num_classes = num_classes
        
        self.output_dir_predictions = os.path.join(output_dir, "predicitions")
        os.makedirs(self.output_dir_predictions, exist_ok=True)
        
        self.distributed = distributed
        self.fp16 = fp16
        #self.testdev = testdev

    #def evaluate(
    #    self, 
    #    model,
    #    half=False,
    #    trt_file=None,
    #    decoder=None,
    #    test_size=None,
    #):
    def evaluate(self, model):
        """
        COCO average precision (AP) Evaluation. Iterate inference on the test dataset
        and the results are evaluated by COCO API.

        NOTE: This function will change training mode to False, please save states if needed.

        Args:
            model : model to evaluate.

        Returns:
            ap50_95 (float) : COCO AP of IoU=50:95
            ap50 (float) : COCO AP of IoU=50
            summary (sr): summary info of evaluation.
        """
        # TODO half to amp_test
        tensor_type = torch.cuda.HalfTensor if self.fp16 else torch.cuda.FloatTensor
        model = model.eval()
        if self.fp16:
            model = model.half()
        ids = []
        data_list = []
        progress_bar = tqdm if is_main_process() else iter

        inference_time = 0
        nms_time = 0
        n_samples = len(self.dataloader) - 1

        # TensorRT stuff
        #if trt_file is not None:
            #from torch2trt import TRTModule
            #model_trt = TRTModule()
            #model_trt.load_state_dict(torch.load(trt_file))
            #x = torch.ones(1, 3, test_size[0], test_size[1]).cuda()
            #model(x)
            #model = model_trt

        for cur_iter, (imgs, _, info_imgs, ids) in enumerate(progress_bar(self.dataloader)):
            with torch.no_grad():
                imgs = imgs.type(tensor_type)

                # skip the the last iters since batchsize might be not enough for batch inference
                is_time_record = cur_iter < len(self.dataloader) - 1
                if is_time_record:
                    start = time.time()

                model_outputs = model(imgs)
                
                #if decoder is not None:
                    #outputs = decoder(outputs, dtype=outputs.type())

                if is_time_record:
                    infer_end = time_synchronized()
                    inference_time += infer_end - start
                
                outputs = postprocess(model_outputs, self.num_classes, self.confthre, self.nmsthre)
    
                if is_time_record:
                    nms_end = time_synchronized()
                    nms_time += nms_end - infer_end

            data_list.extend(self.convert_to_coco_format(outputs, info_imgs, ids))
        
        statistics = torch.tensor([inference_time, nms_time, n_samples], dtype=torch.float, device='cuda:0')
        if self.distributed:
            data_list = gather(data_list, dst=0)
            data_list = list(itertools.chain(*data_list))
            torch.distributed.reduce(statistics, dst=0)

        eval_results = self.evaluate_prediction(data_list, statistics)
        synchronize()
        return eval_results


    def convert_to_coco_format(self, outputs, info_imgs, ids):
        data_list = []
        for (output, img_h, img_w, img_id) in zip(outputs, info_imgs[0], info_imgs[1], ids):
            if output is None:
                continue
            output = output.cpu()
            bboxes = output[:, 0:4]

            # preprocessing: resize
            scale = min(self.img_size[0] / float(img_h), self.img_size[1] / float(img_w))
            bboxes /= scale
            bboxes = xyxy2xywh(bboxes)

            cls = output[:, 6]
            scores = output[:, 4] * output[:, 5]
            for ind in range(bboxes.shape[0]):
                label = self.dataloader.dataset.class_ids[int(cls[ind])]
                pred_data = {
                    "image_id": int(img_id),
                    "category_id": label,
                    "bbox": bboxes[ind].numpy().tolist(),
                    "score": scores[ind].numpy().item(),
                    "segmentation": [],
                }  # COCO json format
                data_list.append(pred_data)
        return data_list


    def evaluate_prediction(self, data_dict, statistics):
        if not is_main_process():
            return 0, 0, None

        logger.info("Evaluate in main process...")

        annType = ["segm", "bbox", "keypoints"]

        inference_time = statistics[0].item()
        nms_time = statistics[1].item()
        n_samples = statistics[2].item()

        a_infer_time = 1000 * inference_time / (n_samples * self.dataloader.batch_size)
        a_nms_time = 1000 * nms_time / (n_samples * self.dataloader.batch_size)

        time_info = ", ".join(
            [
                "Average {} time: {:.2f} ms".format(k, v)
                for k, v in zip(
                    ["forward", "NMS", "inference"],
                    [a_infer_time, a_nms_time, (a_infer_time + a_nms_time)],
                )
            ]
        )

        info = time_info + "\n"

        # Evaluate the Dt (detection) json comparing with the ground truth
        if len(data_dict) > 0:
            cocoGt = self.dataloader.dataset.coco
            coco_predictions_json = os.path.join(self.output_dir_predictions, f'conf{self.confthre}_nms{self.nmsthre}.json')
            with open(coco_predictions_json, 'w') as f:
                json.dump(data_dict, f)
            cocoDt = cocoGt.loadRes(coco_predictions_json)

            #from pycocotools.cocoeval import COCOeval
            from yolox.layers import COCOeval_opt as COCOeval
            cocoEval = COCOeval(cocoGt, cocoDt, annType[1])
            cocoEval.evaluate()
            cocoEval.accumulate()
            redirect_string = io.StringIO()
            with contextlib.redirect_stdout(redirect_string):
                cocoEval.summarize()
            info += redirect_string.getvalue()
            self.save_results(cocoEval)
            return cocoEval.stats[0], cocoEval.stats[1], info
        else:
            return 0, 0, info
        
    
    def save_results(self, cocoEvalObj):
        #metric_IoU_area_maxDets
        header = ["AP_0.50:0.95_all_100", "AP_0.50_all_100", "AP_0.75_all_100", 
                  "AP_0.50:0.95_small_100", "AP_0.50:0.95_medium_100", "AP_0.50:0.95_large_100",
                  "AR_0.50:0.95_all_1", "AR_0.50:0.95_all_10", "AR_0.50:0.95_all_100",
                  "AR_0.50:0.95_small_100", "AR_0.50:0.95_medium_100", "AR_0.50:0.95_large_100"
                ]
        scores = cocoEvalObj.stats

        scores_csv = os.path.join(self.output_dir_predictions, f'conf{self.confthre}_nms{self.nmsthre}.csv')

        with open(scores_csv, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(header)
            writer.writerow(scores)