#from collections import defaultdict
from loguru import logger
from tqdm import tqdm
#import contextlib
#import io
import os
import glob
from collections import OrderedDict
import itertools
#import json
#import tempfile
import time
import torch
import motmetrics as mm
from pathlib import Path

from yolox.utils import (
    gather,
    is_main_process,
    #postprocess,
    synchronize,
    time_synchronized,
    #xyxy2xywh
)
from yolox.data import get_yolox_datadir
#from yolox.tracker.byte_tracker import BYTETracker
#from yolox.sort_tracker.sort import Sort
#from yolox.deepsort_tracker.deepsort import DeepSort
#from yolox.motdt_tracker.motdt_tracker import OnlineTracker

from eval.coco_evaluator import COCOEvaluator
from shared_utils.utils import postprocess
from tracker.byte_tracker_modified import BYTETracker


class MOTEvaluator(COCOEvaluator):
    #def __init__(self, args, dataloader, img_size, confthre, nmsthre, num_classes):
    def __init__(self, dataloader, img_size, confthre, nmsthre, num_classes, output_dir, track_thresh, track_buffer, match_thresh, min_box_area, distributed=False, fp16=False):
        """
        Args:
            dataloader (Dataloader): evaluate dataloader.
            img_size (int): image size after preprocess. images are resized
                to squares whose shape is (img_size, img_size).
            confthre (float): confidence threshold ranging from 0 to 1, which
                is defined in the config file.
            nmsthre (float): IoU threshold of non-max supression ranging from 0 to 1.
        """
        # initiate COCOEvaluator
        super().__init__(
            dataloader=dataloader, 
            img_size=img_size, 
            confthre=confthre, 
            nmsthre=nmsthre, 
            num_classes=num_classes, 
            output_dir=output_dir, 
            distributed=distributed, 
            fp16=fp16
        )
        # initiate MOTEvaluator
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.min_box_area = min_box_area
        
        self.output_dir_tracking = os.path.join(output_dir, "tracking_tracks")
        os.makedirs(self.output_dir_tracking, exist_ok=True)

        self.output_dir_tracking_metrics = os.path.join(output_dir, "tracking_metrics")
        os.makedirs(self.output_dir_tracking_metrics, exist_ok=True)
        #self.dataloader = dataloader
        #self.img_size = img_size
        #self.confthre = confthre
        #self.nmsthre = nmsthre
        #self.num_classes = num_classes

        
    #def evaluate(
    #    self,
    #    model,
    #    distributed=False,
    #    half=False,
    #    trt_file=None,
    #    decoder=None,
    #    test_size=None,
    #    result_folder=None
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
        if model.training:
            logger.info("Setting model to the evaluation model")
            model = model.eval()
        if self.fp16:
            #logger.info("Setting model to half()")
            model = model.half()
        ids = []
        data_list = []
        progress_bar = tqdm if is_main_process() else iter

        results = []
        video_names = {}
        inference_time = 0
        track_time = 0
        n_samples = len(self.dataloader) - 1

        #if trt_file is not None:
            #from torch2trt import TRTModule
            #model_trt = TRTModule()
            #model_trt.load_state_dict(torch.load(trt_file))
            #x = torch.ones(1, 3, test_size[0], test_size[1]).cuda()
            #model(x)
            #model = model_trt
            
        #tracker = BYTETracker(self.args) # TODO adapt
        #ori_thresh = self.track_thresh
        for cur_iter, (imgs, _, info_imgs, ids) in enumerate(progress_bar(self.dataloader)):
            with torch.no_grad():
                # init tracker
                frame_id = info_imgs[2].item()
                video_id = info_imgs[3].item()
                img_file_name = info_imgs[4]
                video_name = img_file_name[0].split('/')[0]
                #if video_name == 'MOT17-05-FRCNN' or video_name == 'MOT17-06-FRCNN':
                #    self.args.track_buffer = 14
                #elif video_name == 'MOT17-13-FRCNN' or video_name == 'MOT17-14-FRCNN':
                #    self.args.track_buffer = 25
                #else:
                #    self.args.track_buffer = 30

                #if video_name == 'MOT17-01-FRCNN':
                #    self.args.track_thresh = 0.65
                #elif video_name == 'MOT17-06-FRCNN':
                #    self.args.track_thresh = 0.65
                #elif video_name == 'MOT17-12-FRCNN':
                #    self.args.track_thresh = 0.7
                #elif video_name == 'MOT17-14-FRCNN':
                #    self.args.track_thresh = 0.67
                #elif video_name in ['MOT20-06', 'MOT20-08']:
                #    self.args.track_thresh = 0.3
                #else:
                #    self.args.track_thresh = ori_thresh

                if video_name not in video_names:
                    video_names[video_id] = video_name
                if frame_id == 1: 
                    # Initiate a new tracker instance whenever a new video sequence starts
                    tracker = BYTETracker(
                        track_thresh = self.track_thresh, 
                        match_thresh = self.match_thresh,
                        track_buffer = self.track_buffer
                    )
                    # save tracking results before clearing out 
                    if len(results) != 0: 
                        result_filename = os.path.join(self.output_dir_tracking, '{}.txt'.format(video_names[video_id - 1]))
                        self.write_tracks(result_filename, results)
                        results = []

                imgs = imgs.type(tensor_type)

                # skip the the last iters since batchsize might be not enough for batch inference
                is_time_record = cur_iter < len(self.dataloader) - 1
                if is_time_record:
                    start = time.time()

                outputs = model(imgs)
                
                #if decoder is not None:
                    #outputs = decoder(outputs, dtype=outputs.type())

                outputs = postprocess(outputs, self.num_classes, self.confthre, self.nmsthre)
            
                if is_time_record:
                    infer_end = time_synchronized()
                    inference_time += infer_end - start

            output_results = self.convert_to_coco_format(outputs, info_imgs, ids)
            data_list.extend(output_results)

            # run tracking
            if outputs[0] is not None:
                online_targets = tracker.update(outputs[0], info_imgs, self.img_size)
                online_tlwhs = []
                online_ids = []
                online_scores = []
                for t in online_targets:
                    tlwh = t.tlwh
                    tid = t.track_id
                    #vertical = tlwh[2] / tlwh[3] > 1.6
                    #if tlwh[2] * tlwh[3] > self.min_box_area and not vertical:
                    if tlwh[2] * tlwh[3] > self.min_box_area:
                        online_tlwhs.append(tlwh)
                        online_ids.append(tid)
                        online_scores.append(t.score)
                # save results
                results.append((frame_id, online_tlwhs, online_ids, online_scores))

            if is_time_record:
                track_end = time_synchronized()
                track_time += track_end - infer_end
            
            if cur_iter == len(self.dataloader) - 1:
                result_filename = os.path.join(self.output_dir_tracking, '{}.txt'.format(video_names[video_id]))
                self.write_tracks(result_filename, results)

        #statistics = torch.cuda.FloatTensor([inference_time, track_time, n_samples])
        statistics = torch.tensor([inference_time, track_time, n_samples], dtype=torch.float, device='cuda:0')
        if self.distributed:
            data_list = gather(data_list, dst=0)
            data_list = list(itertools.chain(*data_list))
            torch.distributed.reduce(statistics, dst=0)

        eval_results = self.evaluate_prediction(data_list, statistics)
        synchronize()
        return eval_results


    # convert_to_coco_format and evaluate_prediction are already defined in the COCO evaluator
    """ 
    def convert_to_coco_format(self, outputs, info_imgs, ids):
        data_list = []
        for (output, img_h, img_w, img_id) in zip(
            outputs, info_imgs[0], info_imgs[1], ids
        ):
            if output is None:
                continue
            output = output.cpu()

            bboxes = output[:, 0:4]

            # preprocessing: resize
            scale = min(
                self.img_size[0] / float(img_h), self.img_size[1] / float(img_w)
            )
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
        track_time = statistics[1].item()
        n_samples = statistics[2].item()

        a_infer_time = 1000 * inference_time / (n_samples * self.dataloader.batch_size)
        a_track_time = 1000 * track_time / (n_samples * self.dataloader.batch_size)

        time_info = ", ".join(
            [
                "Average {} time: {:.2f} ms".format(k, v)
                for k, v in zip(
                    ["forward", "track", "inference"],
                    [a_infer_time, a_track_time, (a_infer_time + a_track_time)],
                )
            ]
        )

        info = time_info + "\n"

        # Evaluate the Dt (detection) json comparing with the ground truth
        if len(data_dict) > 0:
            cocoGt = self.dataloader.dataset.coco
            # TODO: since pycocotools can't process dict in py36, write data to json file.
            _, tmp = tempfile.mkstemp()
            json.dump(data_dict, open(tmp, "w"))
            cocoDt = cocoGt.loadRes(tmp)
            '''
            try:
                from yolox.layers import COCOeval_opt as COCOeval
            except ImportError:
                from pycocotools import cocoeval as COCOeval
                logger.warning("Use standard COCOeval.")
            '''
            #from pycocotools.cocoeval import COCOeval
            from yolox.layers import COCOeval_opt as COCOeval
            cocoEval = COCOeval(cocoGt, cocoDt, annType[1])
            cocoEval.evaluate()
            cocoEval.accumulate()
            redirect_string = io.StringIO()
            with contextlib.redirect_stdout(redirect_string):
                cocoEval.summarize()
            info += redirect_string.getvalue()
            return cocoEval.stats[0], cocoEval.stats[1], info
        else:
            return 0, 0, info
    """

    def write_tracks(self, filename, results, no_score = False):
        if no_score:
            save_format = '{frame},{id},{x1},{y1},{w},{h},-1,-1,-1,-1\n'
        else:
            save_format = '{frame},{id},{x1},{y1},{w},{h},{s},-1,-1,-1\n'
        
        with open(filename, 'w') as f:
            for frame_id, tlwhs, track_ids, scores in results:
                for tlwh, track_id, score in zip(tlwhs, track_ids, scores):
                    if track_id < 0:
                        continue
                    x1, y1, w, h = tlwh
                    line = save_format.format(frame=frame_id, id=track_id, x1=round(x1, 1), y1=round(y1, 1), w=round(w, 1), h=round(h, 1), s=round(score, 2))
                    f.write(line)
        logger.info('save results to {}'.format(filename))

    """
    def write_tracks_no_score(self, filename, results):
        save_format = '{frame},{id},{x1},{y1},{w},{h},-1,-1,-1,-1\n'
        with open(filename, 'w') as f:
            for frame_id, tlwhs, track_ids in results:
                for tlwh, track_id in zip(tlwhs, track_ids):
                    if track_id < 0:
                        continue
                    x1, y1, w, h = tlwh
                    line = save_format.format(frame=frame_id, id=track_id, x1=round(x1, 1), y1=round(y1, 1), w=round(w, 1), h=round(h, 1))
                    f.write(line)
        logger.info('save results to {}'.format(filename))
    """
    
    def evaluate_tracking(self):
        # evaluate MOTA
        mm.lap.default_solver = 'lap'

        gtfiles = [os.path.join(get_yolox_datadir(), "multi_view_mot", "test", seq, "gt", "gt.txt") for seq in os.listdir(os.path.join(get_yolox_datadir(), "multi_view_mot", "test"))]
        tsfiles = [f for f in glob.glob(os.path.join(self.output_dir_tracking, '*.txt')) if not os.path.basename(f).startswith('eval')]
        logger.info('Found {} groundtruths and {} test files.'.format(len(gtfiles), len(tsfiles)))
        logger.info(f'Ground truth {gtfiles}')
        logger.info(f'Tracks {tsfiles}')
        logger.info('Available LAP solvers {}'.format(mm.lap.available_solvers))
        logger.info('Default LAP solver \'{}\''.format(mm.lap.default_solver))
        logger.info('Loading files.')

        gt = OrderedDict([(Path(f).parts[-3], mm.io.loadtxt(f, fmt='mot15-2D', min_confidence=1)) for f in gtfiles])
        ts = OrderedDict([(os.path.splitext(Path(f).parts[-1])[0], mm.io.loadtxt(f, fmt='mot15-2D', min_confidence=-1)) for f in tsfiles])    
        
        mh = mm.metrics.create()    
        accs, names = self.__compare_dataframes(gt, ts)
        
        logger.info('Running metrics')
        metrics = mm.metrics.motchallenge_metrics + ['num_objects']
        summary = mh.compute_many(accs, names=names, metrics=metrics, generate_overall=True)
        
        # convert absolute metrics to relative metrics
        fmt = mh.formatters
        div_dict = {
            'num_objects': ['num_false_positives', 'num_misses', 'num_switches', 'num_fragmentations'],
            'num_unique_objects': ['mostly_tracked', 'partially_tracked', 'mostly_lost']}
        for divisor in div_dict:
            for divided in div_dict[divisor]:
                summary[divided] = (summary[divided] / summary[divisor])
        change_fmt_list = ['num_false_positives', 'num_misses', 'num_switches', 'num_fragmentations', 'mostly_tracked',
                        'partially_tracked', 'mostly_lost']
        for k in change_fmt_list:
            fmt[k] = fmt['mota']
        
        return mm.io.render_summary(summary, formatters=fmt, namemap=mm.io.motchallenge_metric_names)

    
    def __compare_dataframes(self, gts, ts):
        accs = []
        names = []
        for k, tsacc in ts.items():
            if k in gts:            
                logger.info('Comparing {}...'.format(k))
                accs.append(mm.utils.compare_to_groundtruth(gts[k], tsacc, 'iou', distth=0.5))
                names.append(k)
            else:
                logger.warning('No ground truth for {}, skipping.'.format(k))
        return accs, names
