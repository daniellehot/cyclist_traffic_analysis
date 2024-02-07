from detectron2.data import transforms as T
from detectron2.data import DatasetCatalog
import numpy as np

#TODO Logic to ensure test resize scale and train resize scale results in similar image dimensions if jai and rs datasets are used. Currently, a user is expected to compute the scale
def get_augmentations(configuration_file):
    #train augmentations
    train_dataset = DatasetCatalog.get(configuration_file['train_dataset'])
    train_img_height, train_img_width = train_dataset[0]['height'], train_dataset[0]['width']
    train_augmentations = _create_augmentations_list(configuration_file['train_augmentations'], train_img_height, train_img_width)
    
    #test augmentations
    test_dataset = DatasetCatalog.get(configuration_file['test_dataset'])
    test_img_height, test_img_width = test_dataset[0]['height'], test_dataset[0]['width']
    test_augmentations = _create_augmentations_list(configuration_file['test_augmentations'], test_img_height, test_img_width)
    return train_augmentations, test_augmentations


def _create_augmentations_list(yaml_augs, img_height, img_width):
    augs = []
    
    if yaml_augs.get('hFlip') and yaml_augs['hFlip']:
        augs.append(T.RandomFlip(prob=yaml_augs['hFlip_prob'], horizontal=True, vertical=False))
    
    if yaml_augs.get('vFlip') and yaml_augs['vFlip']: 
        augs.append(T.RandomFlip(prob=yaml_augs['vFlip_prob'], horizontal=False, vertical=True))

    if yaml_augs.get('rBrightness') and yaml_augs['rBrightness']:
        augs.append(T.RandomBrightness(intensity_min=yaml_augs['bright_min'], intensity_max=yaml_augs['bright_max']))
    
    if yaml_augs.get('rContrast') and yaml_augs['rContrast']:
        augs.append(T.RandomContrast(intensity_min=yaml_augs['cont_min'], intensity_max=yaml_augs['cont_max']))
    
    if yaml_augs.get('rSaturation') and yaml_augs['rSaturation']:
        augs.append(T.RandomSaturation(intensity_min=yaml_augs['sat_min'], intensity_max=yaml_augs['sat_max']))
    
    if yaml_augs.get('resize') and yaml_augs['resize']:
        img_target_shape = (int(img_height*yaml_augs['scale']), int(img_width*yaml_augs['scale'])) 
        augs.append(T.Resize(shape=img_target_shape))
    
    if yaml_augs.get('rCrop') and yaml_augs['rCrop']['crop']:
        augs.append(T.RandomCrop(yaml_augs['rCrop']['type'], crop_size=tuple(yaml_augs['rCrop']['crop_size'])))

    return augs
