import copy


class ValidationLoss(HookBase):
    """
    A hook that computes validation loss during training.

    Attributes:
        cfg (CfgNode): The detectron2 config node.
        _loader (iterator): An iterator over the validation dataset.
    """
    '''
    def __init__(self, cfg):
        """
        Args:
            cfg (CfgNode): The detectron2 config node.
        """
        super().__init__()
        self.cfg = cfg.clone()
        # Switch to the validation dataset
        self.cfg.DATASETS.TRAIN = cfg.DATASETS.VAL
        # Build the validation data loader iterator
        self._loader = iter(build_detection_train_loader(self.cfg))
    '''

    def __init__(self, cfg):
        super().__init__()
        self.cfg = copy.deepcopy(cfg)
        self.cfg.dataloader.train.dataset.names = cfg.dataloader.test.dataset.names
        self._loader = iter(instantiate(self.cfg.dataloader.train))
        

    def after_step(self):
        """
        Computes the validation loss after each training step.
        """
        # Get the next batch of data from the validation data loader
        data = next(self._loader)
        with torch.no_grad():
            # Compute the validation loss on the current batch of data
            loss_dict = self.trainer.model(data)

            # Check for invalid losses
            losses = sum(loss_dict.values())
            assert torch.isfinite(losses).all(), loss_dict

            # Reduce the loss across all workers
            loss_dict_reduced = {"val_" + k: v.item() for k, v in
                                 comm.reduce_dict(loss_dict).items()}
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())

            # Save the validation loss in the trainer storage
            if comm.is_main_process():
                self.trainer.storage.put_scalars(total_val_loss=losses_reduced,
                                                 **loss_dict_reduced)
                
"""
class ImageSaver(HookBase):
    def __init__(self, cfg, dataset_classes, data_augmentations, log_period=0, output_dir="/workspace/sample_training_images"):
        self.cfg = cfg.clone()
        #self.data_loader = iter(get_train_loader(self.cfg))
        self.data_augmentations = data_augmentations
        # Build the data loader iterator
        self.data_loader = build_detection_train_loader(
            self.cfg, 
            mapper = COCOInstanceNewBaselineDatasetMapper(
                is_train=True,
                image_format=cfg.INPUT.FORMAT,
                tfm_gens = self.data_augmentations
            ) 
        )
        self.data_loader_itr = iter(self.data_loader)
        
        #self.metadata = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]) 
        self.dataset_classes = dataset_classes
        self.sample_period = log_period 
        if self.sample_period > 0 and os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def after_step(self):
        # Check if it's time to save an image sample
        if self.sample_period > 0 and self.trainer.iter % self.sample_period == 0:
            batch = next(self.data_loader_itr)   
            for sample in batch:
                id = str(uuid.uuid4())
                img = sample['image'].to("cpu").numpy()
                img = np.moveaxis(img, 0, -1)
                boxes = sample['instances'].get('gt_boxes')
                masks = sample['instances'].get('gt_masks')
                labels = self.ids_to_names(sample['instances'].get('gt_classes'))
                #v = Visualizer(img, metadata=self.metadata)
                v = Visualizer(img)
                v = v.overlay_instances(
                    boxes=boxes, 
                    masks=masks,
                    labels=labels
                )
                img_annotated = v.get_image()
                concatenated_img = cv2.hconcat([img, img_annotated])
                concatenated_img = cv2.cvtColor(concatenated_img, cv2.COLOR_BGR2RGB)
                cv2.imwrite(
                    os.path.join(self.output_dir, id+".png"),
                    concatenated_img
                )
        
    def ids_to_names(self, ids):
        names = [self.dataset_classes[id] for id in ids]
        return names
"""