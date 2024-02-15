from pycocotools.coco import COCO
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Script description")
    parser.add_argument("-i", "--input", type=str, help="Annotation file")
    parser.add_argument("-o", "--output", type=str, help="Where to save the text file")
    return parser.parse_args()


def main(args):
    coco = COCO(args.input)
    category_ids = coco.getCatIds()
    categories = coco.loadCats(category_ids)
    category_names = [category['name'] for category in categories]
    
    with open(f"{args.output}/categories.txt", 'w') as f:
        for name in category_names:
            f.write("%s\n" % name)


if __name__=="__main__":
    main(parse_args())
