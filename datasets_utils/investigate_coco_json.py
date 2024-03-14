from pycocotools.coco import COCO
import pandas as pd
import argparse 


def parse_args():
    parser = argparse.ArgumentParser(description="COCO json explorer")
    parser.add_argument("-i", "--input", type=str, help="Annotation file")
    return parser.parse_args()


if __name__=="__main__":
    args = parse_args()
    coco = COCO(args.input)
    
    images_coco = coco.loadImgs(coco.getImgIds())
    images_df = pd.DataFrame(images_coco)
    print("=========\nImages COCO dictionary\n=========")
    print(images_df)
    
    annotations_coco = coco.loadAnns(coco.getAnnIds())
    annotations_df = pd.DataFrame(annotations_coco)
    print("=========\nAnnotations COCO dictionary\n=========")
    print(annotations_df)

    categories_coco = coco.loadCats(coco.getCatIds())
    categories_df = pd.DataFrame(categories_coco)
    print("=========\nCategories COCO dictionary\n=========")
    print(categories_df)

    # Count instances per category
    # First, add a column with category names to use instead of category ids
    category_id_to_name = {category['id']: category['name'] for category in categories_coco}
    annotations_df['category_name'] = annotations_df['category_id'].map(category_id_to_name)
    category_counts = annotations_df['category_name'].value_counts()
    print("=========\nInstances per Category\n=========")
    print(category_counts)

    print(type(images_coco), images_coco[0].keys(), images_coco[0])
    print(type(annotations_coco), annotations_coco[0].keys(), annotations_coco[0])
    print(type(categories_coco), categories_coco[0].keys(), categories_coco[0])

