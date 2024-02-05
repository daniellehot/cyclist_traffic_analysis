# common
import os
import argparse
import yaml
from omegaconf import OmegaConf

# detectron2 
import logging
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import instantiate, LazyConfig
from detectron2.engine import (
    AMPTrainer,
    default_argument_parser,
    default_setup,
    default_writers,
    hooks,
    launch,
    SimpleTrainer,
)
from detectron2.engine.defaults import create_ddp_model
from detectron2.evaluation import inference_on_dataset, print_csv_format
from detectron2.utils import comm
from detectron2.utils.logger import setup_logger

# shared
from register_generic_dataset import TrafficDataset
import augmentations_utils


#logger = logging.getLogger("detectron2")

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-y', '--yaml', type=str, help="Path to a yaml experiment file.")
    parser.add_argument('-r', '--resume', action='store_true', help="Resume from a checkpoint.")
    parser.add_argument('--checkpoint', type=str, help="Checkpoint to resume from.")
    args = parser.parse_args()
    return args


def get_config(experiment):
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

    train_augmentations, test_augmentations = augmentations_utils.get_augmentations(experiment)
    cfg.dataloader.train.mapper.augmentations = train_augmentations
    cfg.dataloader.test.mapper.augmentations = test_augmentations

    cfg.dataloader.train.num_workers = experiment['num_of_workers'] 
    cfg.dataloader.test.num_workers = experiment['num_of_workers']
    cfg.dataloader.train.total_batch_size = experiment['batch_size']

    # Validation setup (same as train except for the dataset name and augmentations)
    cfg.dataloader.validation = cfg.dataloader.train
    cfg.dataloader.validation.dataset.names = experiment['val_dataset']
    cfg.dataloader.validation.mapper.augmentations = test_augmentations

    cfg.model.roi_heads.num_classes = len(experiment['classes'])
    # save experiment model configuration
    OmegaConf.save(cfg, os.path.join(experiment['output_dir'], "swin_base_experiment.yaml"))
    return cfg


# TODO Implement resuming from checkpoint
# TODO Validation Hook, Image Saver Hook
def train(configuration, resume=False, checkpoint=None):
    model = instantiate(configuration.model)
    #logger = logging.getLogger("detectron2")
    #logger.info("Model:\n{}".format(model))
    model.to(configuration.train.device)

    configuration.optimizer.params.model = model
    optimizer = instantiate(configuration.optimizer)

    train_loader = instantiate(configuration.dataloader.train)

    model = create_ddp_model(model, **configuration.train.ddp)
    trainer = (AMPTrainer if configuration.train.amp.enabled else SimpleTrainer)(model, train_loader, optimizer)
    checkpointer = DetectionCheckpointer(
        model,
        configuration.train.output_dir,
        trainer=trainer,
    )

    writer = default_writers(configuration.train.output_dir, configuration.train.max_iter)

    trainer.register_hooks([
        hooks.IterationTimer(),
        hooks.LRScheduler(scheduler=instantiate(configuration.lr_multiplier)),
        hooks.PeriodicCheckpointer(checkpointer, **configuration.train.checkpointer) if comm.is_main_process() else None,
        #hooks.EvalHook(cfg.train.eval_period, lambda: test(cfg, model)),
        hooks.PeriodicWriter( writer, period=configuration.train.log_period) if comm.is_main_process() else None,
    ])
    
    trainer.train(0, configuration.train.max_iter)
    

    # The checkpoint stores the training iteration that just finished, thus we start
    # at the next iteration
    #    start_iter = trainer.iter + 1
    #else:
    #    start_iter = 0
   # trainer.train(start_iter, cfg.train.max_iter)
    

def main(args):
    # Read the experiment file
    with open(args.yaml, "r") as f:
        experiment_configuration = yaml.safe_load(f)
    # Create output directory 
    os.makedirs(experiment_configuration['output_dir'], exist_ok=True)

    setup_logger(output=experiment_configuration['output_dir'], distributed_rank=comm.get_rank(), name="cowi_traffic_analysis")

    # dataset
    dataset = TrafficDataset(experiment_configuration['dataset_folder'])
    dataset.register_dataset()

    # configuration
    configuration = get_config(experiment_configuration)
    
    #training
    train(configuration, resume=args.resume, checkpoint=args.checkpoint)
    

if __name__=="__main__":
    main(parse_arguments())