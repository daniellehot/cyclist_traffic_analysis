from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader, DatasetMapper, get_detection_dataset_dicts    
import augmentations_utils

def get_COCO_evaluator(configuration):
    evaluator = COCOEvaluator(
        dataset_name=configuration.dataloader.evaluator.dataset_name[0],
        tasks="bbox",
        output_dir=configuration.dataloader.evaluator.output_dir, 
        use_fast_impl=False,
        allow_cached_coco=False)
    return evaluator
    

def get_test_data_loader(configuration):
    data_loader = build_detection_test_loader(
        dataset = get_detection_dataset_dicts(configuration.dataloader.test.dataset.names),
        #total_batch_size = configuration.dataloader.test.total_batch_size,
        batch_size = 1,
        num_workers = configuration.dataloader.test.num_workers,
        mapper = DatasetMapper(
            is_train = False,
            image_format = configuration.dataloader.test.mapper.image_format,
            augmentations = configuration.dataloader.test.mapper.augmentations
        )
    )
    return data_loader


"""
def compute_ap(self, data_loader_batch_size = 1, data_loader_num_worker = 1, task=None, fast=False, output_suffix=None):
    #########################################################################
    #########################################################################
    def compute_AP_for_every_IoU(results):
        print("Computing AP for every IoU threshold")
        gt = os.path.join(self.test_cfg.OUTPUT_DIR, f"{self.test_cfg.DATASETS.TEST[0]}_coco_format.json")
        pred = os.path.join(self.test_cfg.OUTPUT_DIR, "coco_instances_results.json")
        
        coco_gt = COCO(gt)
        predictions = coco_gt.loadRes(pred)
        eval_res = COCOeval(coco_gt, predictions)
        eval_res.evaluate()
        eval_res.accumulate()
        #eval_res.summarize()

        p = eval_res.eval["params"]
        areaRng = 'all'
        maxDets = 100
        aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
        mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
        
        IoUs, APs = [], []
        for iouThr in p.iouThrs:
            if iouThr is not None:
                s = copy.deepcopy(eval_res.eval['precision'])
                t = np.where(iouThr == p.iouThrs)[0]
                s = s[t]
                s = s[:,:,:,aind,mind]
                IoUs.append(np.round(iouThr, 2))
                APs.append(s.mean())
                #print(f"AP@{np.round(iouThr, 2)} {np.round(s.mean(), 3)}")
        
        for IoU, ap in zip(IoUs, APs):
            if int(IoU*100) != 50 or int(IoU*100) != 75:
                results['segm'][f'AP{int(100*IoU)}'] = ap*100
        print(results)
        return results
    #########################################################################
    #########################################################################
    def write_csv(output, data):
        # Modify the structure
        modified_data = {}
        for category, metrics in data.items():
            for key, value in metrics.items():
                modified_data[f"{category}/{key}"] = value

        # Write to CSV
        with open(output, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # Write the headers
            writer.writerow(modified_data.keys())
            # Write the data
            writer.writerow(modified_data.values())
        print(f"Saved results to {output}")  
    #########################################################################
    #########################################################################

    evaluator = COCOEvaluator(self.test_cfg.DATASETS.TEST[0],
                tasks=task,
                output_dir=self.test_cfg.OUTPUT_DIR, 
                use_fast_impl=fast,
                allow_cached_coco=False)
        
    data_loader = build_detection_test_loader(
        self.test_cfg,
        self.test_cfg.DATASETS.TEST[0],
        mapper = DatasetMapper(
            self.test_cfg, 
            is_train=False, 
            augmentations=self.data_augmentations
        ),
        batch_size = data_loader_batch_size,
        num_workers = data_loader_num_worker
    )
        
    with autocast():
        results = inference_on_dataset(self.predictor.model, data_loader, evaluator)

    results = compute_AP_for_every_IoU(results)
    output_filename = "ap_{}.csv".format(self.test_cfg.DATASETS.TEST[0])
    
    if output_suffix is not None:
        output_filename = "ap_{}_{}.csv".format(self.test_cfg.DATASETS.TEST[0], output_suffix)
        

    write_csv(output=os.path.join(self.test_cfg.OUTPUT_DIR, output_filename), data=results)  
"""