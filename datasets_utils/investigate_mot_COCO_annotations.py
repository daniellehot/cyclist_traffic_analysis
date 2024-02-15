from pycocotools.coco import COCO
import argparse
import os
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description="Script description")
    parser.add_argument("-i", "--input", type=str, help="Annotation file")
    return parser.parse_args()


def main(args):
    coco = COCO(args.input)
    category_ids = coco.getCatIds()
    categories = coco.loadCats(category_ids)
    
    # Print all category names
    print("All category names:")
    for category in categories:
        print(category['name'])
    print(f"Number of categories {len(categories)}")

    # histogram
    # Initialize a dictionary to store the count of annotations per class
    annotations_per_class = {category['name']: 0 for category in categories}

    # Get all annotations
    annotation_ids = coco.getAnnIds()
    annotations = coco.loadAnns(annotation_ids)

    # Count annotations per class
    for annotation in annotations:
        category_id = annotation['category_id']
        category_name = coco.loadCats(category_id)[0]['name']
        annotations_per_class[category_name] += 1
    
    # Remove classes with zero annotations
    annotations_per_class = {k: v for k, v in annotations_per_class.items() if v > 0}

    # Plot histogram
    plt.figure(figsize=(10, 6))
    plt.bar(annotations_per_class.keys(), annotations_per_class.values())
    plt.xlabel('Class')
    plt.ylabel('Number of Annotations')
    plt.title('Histogram of Annotations per Class')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig("multi_view_dataset_hist.png")


if __name__=="__main__":
    main(parse_args())