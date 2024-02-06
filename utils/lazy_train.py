# common
import os, shutil
import argparse
import yaml
from torch.cuda.amp import autocast
import logging

# detectron2 
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import instantiate
from detectron2.engine import (
    #default_argument_parser, #replaced with my arguments
    #default_setup, #replaced with my create_log()
    launch, #used for distributed training 
    default_writers,
    hooks,
    AMPTrainer,
    SimpleTrainer,
)
from detectron2.engine.defaults import create_ddp_model
from detectron2.utils import comm
from detectron2.evaluation import inference_on_dataset, print_csv_format

# shared
from register_generic_dataset import TrafficDataset
from train_utils import (
    save_training_images, 
    ValidationLoss, 
    plot_metrics_json, 
    create_log,
    get_config
)
from test_utils import get_COCO_evaluator, get_test_data_loader


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-y', '--yaml', type=str, help="Path to a yaml experiment file.")
    parser.add_argument('--eval-only', action='store_true', help="Run only evaluation")
    parser.add_argument('--save-training-images', action='store_true', help="Save a sample of training images")
    parser.add_argument('--sample-size', type=int, default=10, help="How many train images to save. Default is 10")
    parser.add_argument('--show-annotations', action='store_true', help="Overlay train images with annotations. Default is False")
    #parser.add_argument('-r', '--resume', action='store_true', help="Resume from a checkpoint.")
    #parser.add_argument('--checkpoint', type=str, help="Checkpoint to resume from.")
    args = parser.parse_args()
    return args



def eval(configuration, weights_to_evaluate):
    logger = logging.getLogger("ViTDet")
    logger.info(f"eval(). weight_to_evaluate{weights_to_evaluate}")

    model = instantiate(configuration.model)
    model.to(configuration.train.device)
    model = create_ddp_model(model)
    DetectionCheckpointer(model).load(weights_to_evaluate)

    test_data_loader = get_test_data_loader(configuration)
    print(test_data_loader)
    
    evaluator = get_COCO_evaluator(configuration)
    print(evaluator)

    with autocast():
        results = inference_on_dataset(model, test_data_loader, evaluator)
    
    logger.info(f"Results {results}")
    """
    if "evaluator" in configuration.dataloader:
        ret = inference_on_dataset(
            model,
            instantiate(configuration.dataloader.test),
            instantiate(configuration.dataloader.evaluator),
        )
        print_csv_format(ret)
        return ret
    """

# TODO Implement resuming from checkpoint
def train(configuration, resume=False, checkpoint=None):
    model = instantiate(configuration.model)
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
        ValidationLoss(configuration),
        #hooks.EvalHook(cfg.train.eval_period, lambda: test(cfg, model)),
        hooks.PeriodicWriter(writer, period=configuration.train.log_period) if comm.is_main_process() else None,
    ])
    
    trainer.train(0, configuration.train.max_iter)
    

def main(args):
    # Read the experiment file
    with open(args.yaml, "r") as f:
        experiment_configuration = yaml.safe_load(f)
    
    # Create output directory 
    if os.path.exists(experiment_configuration['output_dir']) and not args.eval_only:
        shutil.rmtree(experiment_configuration['output_dir'])
    else:
        os.makedirs(experiment_configuration['output_dir'], exist_ok=True)

    # Create logger 
    create_log(args=args, output_dir=experiment_configuration['output_dir'])

    # dataset
    dataset = TrafficDataset(experiment_configuration['dataset_folder'])
    dataset.register_dataset()  

    # configuration
    configuration = get_config(experiment_configuration)

    # save some training images
    if args.save_training_images:
        save_training_images(configuration, output_dir=experiment_configuration['output_dir'], sample_size=args.sample_size, show_annotations=args.show_annotations)

    # training/evaluation
    if args.eval_only:
        eval(configuration, os.path.join(experiment_configuration['output_dir'], "model_final.pth"))
    else:
        #train(configuration, resume=args.resume, checkpoint=args.checkpoint)
        train(configuration)
        eval(configuration, os.path.join(experiment_configuration['output_dir'], "model_final.pth"))

    # plot collected metrics
    plot_metrics_json(
        input=os.path.join(experiment_configuration['output_dir'], "metrics.json"),
        output_dir=os.path.join(experiment_configuration['output_dir']),
    )
    

if __name__=="__main__":
    #main(parse_arguments())
    launch(
        main(parse_arguments()), 
        num_gpus_per_machine=1
    )