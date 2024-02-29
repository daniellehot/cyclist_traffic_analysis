import json
import os


ROOT = os.path.expanduser("~/cyclist_traffic_analysis/multi-view-data")
INFRASTRUCTURE = ROOT +"/Sequence3/infrastructure-mscoco.json"
DRONE = ROOT + "/Sequence3/drone-mscoco.json"
INFRASTRUCTURE_PNG = ROOT + "/Sequence3-png/Infrastructure/infrastructure-mscoco.json"
DRONE_PNG = ROOT + "/Sequence3-png/Drone/drone-mscoco.json"


def read_filenames(json_path):
    with open(json_path, 'r') as f:
        annotations = json.load(f)
    filenames = []
    for image in annotations['images']:
        filenames.append(image['file_name'])
    return filenames


if __name__=="__main__":
    infrastructure_files = read_filenames(INFRASTRUCTURE_PNG)
    infrastructure_files = sorted([file.replace("Sequence3-png/Infrastructure/", "") for file in infrastructure_files])
    
    drone_files = read_filenames(DRONE_PNG)
    drone_files = sorted([file.replace("Drone/", "") for file in drone_files])
    same = []
    for f1, f2 in zip(infrastructure_files, drone_files):
        f1 = f1.replace("infra", "view")
        f2 = f2.replace("drone", "view")
        if f1 != f2:
            print(f1)
            print(f2)
            print()


    #filenames1 = read_filenames(INFRASTRUCTURE)
    #filenames2 = read_filenames(INFRASTRUCTURE_PNG)
    #print(len(filenames1))
    #print(len(filenames2))
    #print(len(filenames1)==len(filenames2))

    #filenames1 = read_filenames(DRONE)
    #filenames2 = read_filenames(DRONE_PNG)
    #print(len(filenames1))
    #print(len(filenames2))
    #print(len(filenames1)==len(filenames2))