# common modules
import torch
import numpy as np
import os
import logging
import matplotlib.pyplot as plt
import json
from omegaconf import OmegaConf

# detectron2
from detectron2.engine import HookBase
from detectron2.config import instantiate, LazyConfig
from detectron2.utils import comm
from detectron2.data import build_detection_train_loader, DatasetMapper, get_detection_dataset_dicts
from detectron2.utils.visualizer import Visualizer
# create_log() required functions
from detectron2.utils.collect_env import collect_env_info
from detectron2.utils.logger import setup_logger

# shared
import augmentations_utils


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
        # self.cfg = copy.deepcopy(cfg)
        # elf.cfg.dataloader.train.dataset.names = cfg.dataloader.test.dataset.names
        logger = logging.getLogger("ViTDet")
        logger.info(f"Building a validation dataloader for the ValidationLoss hook")
        self._loader = iter(instantiate(cfg.dataloader.validation))
        

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


# TODO save testing/validation images
def save_training_images(configuration, output_dir, sample_size=10, show_annotations=True):
    # create output folder
    output_dir = os.path.join(output_dir, "training_images")
    os.makedirs(output_dir, exist_ok=True)

    logger = logging.getLogger("ViTDet")
    logger.info(f"Saving a sample of train images. output_dir {output_dir} sample_size {sample_size} show_annotations {show_annotations}")

    # create dataloader
    train_loader = build_detection_train_loader(
        dataset = get_detection_dataset_dicts(configuration.dataloader.train.dataset.names),
        total_batch_size = 1,
        num_workers = configuration.dataloader.train.num_workers,
        mapper = DatasetMapper(
            is_train = True,
            image_format = configuration.dataloader.train.mapper.image_format,
            augmentations = configuration.dataloader.train.mapper.augmentations
        )
    )
    data_itr = iter(train_loader)
    
    # visualize and save images
    for sample in range(sample_size):
        batch = next(data_itr)[0]
        img = batch['image'].to("cpu").numpy()
        img = np.moveaxis(img, 0, -1)
        v = Visualizer(img)
        if show_annotations:
            boxes = batch['instances'].get('gt_boxes')
            labels = batch['instances'].get('gt_classes') # labels = self.ids_to_names(sample['instances'].get('gt_classes'))
            v = v.overlay_instances(boxes=boxes, labels=labels)
        else:
            v = v.overlay_instances()
        output_file = os.path.join(output_dir, f"sample_{sample + 1}.png")
        v.save(output_file)


def plot_metrics_json(input, output_dir):
    # create output folder
    output_dir = os.path.join(output_dir, "plots")
    os.makedirs(output_dir, exist_ok=True)

    logger = logging.getLogger("ViTDet")
    logger.info(f"Plotting collected metrics. output_dir {output_dir}")

    # Read the JSON file and handle multiple separate JSON objects
    metrics_list = []
    with open(input, 'r') as f:
        for line in f:
            metrics_list.append(json.loads(line))

    # Combine individual dictionaries into a single dictionary
    combined_metrics = {}
    for entry in metrics_list:
        for key, value in entry.items():
            combined_metrics.setdefault(key, []).append(value)

    # define losses
    losses = [
        "loss_box_reg_stage0",
        "loss_box_reg_stage1",
        "loss_box_reg_stage2",
        "loss_cls_stage0",
        "loss_cls_stage1",
        "loss_cls_stage2",
        #"loss_mask",
        "loss_rpn_cls",
        "loss_rpn_loc",
    ]

    val_losses = [f"val_{loss}" for loss in losses]
    losses.append("total_loss")
    val_losses.append("total_val_loss")
    
    val_losses_filtered = [val_loss if val_loss in combined_metrics.keys() else None for val_loss in val_losses]
    metric_pairs = [(loss, val_loss) for loss, val_loss in zip(losses, val_losses_filtered)]
    metric_pairs.append(('lr', None))

    for metric, val_metric in metric_pairs:
        logger.info(f"Plotting {metric} metric")
        plt.figure()
        # Plot metric and annotate the last point with its rounded value
        iterations = combined_metrics['iteration']
        metric_values = combined_metrics[metric]
        plt.plot(iterations, metric_values, '-o', label=metric)
        #plt.annotate(f"{round(metric_values[-1], 3)}", (iterations[-1], metric_values[-1]))

        # Plot validation metric and annotate the last point with its rounded value, if it exists
        if val_metric:
            val_metric_values = combined_metrics[val_metric]
            plt.plot(iterations, val_metric_values, '-o', label=val_metric, alpha=0.7)
            #plt.annotate(f"{round(val_metric_values[-1], 3)}", (iterations[-1], val_metric_values[-1]), alpha=0.7)
        
        plt.xlabel('Iteration')
        plt.ylabel(metric)
        plt.title(metric)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{metric}.png"))
        plt.close()
    

def create_log(args, output_dir, logger_name="ViTDet"):
    """
    Perform some basic common setups at the beginning of a job, including:

    1. Set up the detectron2 logger
    2. Log basic information about environment, cmdline arguments, and config
    3. Backup the config to the output directory

    Args:
        cfg (CfgNode or omegaconf.DictConfig): the full config to be used
        args (argparse.NameSpace): the command line arguments to be logged
    """

    rank = comm.get_rank()
    setup_logger(output_dir, distributed_rank=rank, name=logger_name) #name="fvcore"
    logger = setup_logger(output_dir, distributed_rank=rank)
    logger.info("Rank of current process: {}. World size: {}".format(rank, comm.get_world_size()))
    logger.info("Environment info:\n" + collect_env_info())
    logger.info("Command line arguments: " + str(args))

    # cudnn benchmark has large overhead. It shouldn't be used considering the small size of
    # typical validation set.
    #if not (hasattr(args, "eval_only") and args.eval_only):
        #torch.backends.cudnn.benchmark = _try_get_key(
        #    cfg, "CUDNN_BENCHMARK", "train.cudnn_benchmark", default=False
        #)


def get_config(experiment, add_augmentations=True):
    cfg = LazyConfig.load(experiment['model_definition'])
    cfg.train.init_checkpoint = experiment['weights']
    # save default model configuration
    OmegaConf.save(cfg, os.path.join(experiment['output_dir'], "swin_base_default.yaml"))
    cfg.train.output_dir = experiment['output_dir']
    
    cfg.train.max_iter = experiment['training_iterations']
    cfg.lr_multiplier.scheduler.num_updates = experiment['training_iterations']
    cfg.lr_multiplier.scheduler.milestones = experiment['lr_milestones']
    cfg.train.checkpointer.period = experiment['checkpoint_rate']
    cfg.train.eval_period = experiment['eval_period']

    if experiment['log_period'] != 'default':
        cfg.train.log_period = experiment['log_period']
    if experiment['learning_rate'] != 'default':
        cfg.optimizer.lr = experiment['lr']

    cfg.dataloader.evaluator.output_dir = experiment['output_dir']
    
    cfg.dataloader.train.dataset.names = experiment['train_dataset']
    cfg.dataloader.test.dataset.names = experiment['test_dataset']

    cfg.dataloader.train.num_workers = experiment['num_of_workers'] 
    cfg.dataloader.train.total_batch_size = experiment['batch_size']
    cfg.dataloader.test.num_workers = experiment['num_of_workers']

    cfg.dataloader.validation = cfg.dataloader.train
    cfg.dataloader.validation.dataset.names = experiment['val_dataset']

    if add_augmentations:
        train_augmentations, test_augmentations = augmentations_utils.get_augmentations(experiment)
        cfg.dataloader.train.mapper.augmentations = train_augmentations
        cfg.dataloader.test.mapper.augmentations = test_augmentations
        cfg.dataloader.validation.mapper.augmentations = test_augmentations

    # Do not use masks in mappers
    cfg.dataloader.train.mapper.use_instance_mask = False
    cfg.dataloader.train.mapper.recompute_boxes = False
    cfg.dataloader.validation.mapper.use_instance_mask = False
    cfg.dataloader.validation.mapper.recompute_boxes = False

    # Remove mask predictions 
    cfg.model.roi_heads.mask_in_features = None  # Remove mask-related features
    cfg.model.roi_heads.mask_pooler = None       # Disable mask pooler
    cfg.model.roi_heads.mask_head = None         # Disable mask head

    cfg.model.roi_heads.num_classes = len(experiment['classes'])

    # save experiment model configuration
    OmegaConf.save(cfg, os.path.join(experiment['output_dir'], "swin_base_experiment.yaml"))
    return cfg


def add_augmentations_to_configuration(cfg, experiment):
    train_augmentations, test_augmentations = augmentations_utils.get_augmentations(experiment)
    cfg.dataloader.train.mapper.augmentations = train_augmentations
    cfg.dataloader.test.mapper.augmentations = test_augmentations
    cfg.dataloader.validation.mapper.augmentations = test_augmentations
    return cfg