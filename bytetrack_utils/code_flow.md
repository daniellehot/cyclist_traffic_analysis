entry point is train.py
- inputs are args and an experiment file
- an experiment file is processed through get_exp() from yolox.exp
    - get_exp() creates an instance of class Exp defined in the experiment file 
    - class Exp inherits from the base yolox experiment class MyExp defined in exp/yolox_base.py, which then inherits from the base abstract class in exp/base_exp.py
- launch() from yolox.core starts training
    - launch() provides a way to easily launch distributed training across multiple GPUs and machines using PyTorch, handling the necessary setup and communication between processes.
    - might not be relevant for me given I do not plan on using distributed training
- class Trainer() from yolox core is responsible for training
    - all training functionality is written here
    - TODO add metrics.json writer

train_loader explained 
- train_loader provides a constant stream of batches
- size of a batch equals number of workers 
- each batch contains user defined number of images

 
 training in epoch
 - An epoch refers to one complete pass of the entire training dataset through the learning algorithm. In other words, when all the data samples have been exposed to the neural network for learning patterns, one epoch is said to be completed.