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

# shared
from register_generic_dataset import TrafficDataset


def parse_arguments():
    parser = argparse.ArgumentParser()
    # Define the argument for the YAML file path
    # parser.add_argument('-d', '--dataset_folder', type=str, help="Path to a dataset folder")
    # parser.add_argument('-c', '--config', type=str, help="Path to a LazyConfig file.")
    # parser.add_argument('-w', '--weights', type=str, help="Path to a weight file.")
    parser.add_argument('-y', '--yaml', type=str, help="Path to a yaml file.")
    args = parser.parse_args()
    return args

# TODO augmentations, training iterations, validation hook, 
def get_model_config(experiment):
    cfg = LazyConfig.load(experiment['model_definition'])
    
    cfg.train.init_checkpoint = experiment['weights']
    cfg.train.output_dir = experiment['output_dir']
    #cfg.train.eval_period = 50 
    #cfg.train.max_iter = 200
    #cfg.train.checkpointer.period = 100


    cfg.dataloader.evaluator.output_dir = experiment['output_dir']
    cfg.dataloader.train.dataset.names = experiment['train_dataset']
    #cfg.dataloader.train.mapper.augmentations = augmentations
    cfg.dataloader.test.dataset.names = experiment['test_dataset']
    #cfg.dataloader.test.mapper.augmentations = augmentations

    cfg.dataloader.train.num_workers = experiment['num_of_workers'] 
    cfg.dataloader.test.num_workers = experiment['num_of_workers']
    cfg.dataloader.train.total_batch_size = experiment['batch_size']

    #cfg.optimizer.lr = 0.001

    cfg.model.roi_heads.num_classes = len(experiment['classes'])

    return cfg


def main(args):
    with open(args.yaml, "r") as f:
        experiment_definition = yaml.safe_load(f)

    # dataset
    dataset = TrafficDataset(experiment_definition['dataset_folder'])
    dataset.register_dataset()

    # model configuration 
    model_config = get_model_config(experiment_definition)
    model_description_yaml = OmegaConf.to_yaml(model_config)
    print(model_description_yaml)

if __name__=="__main__":
    main(parse_arguments())