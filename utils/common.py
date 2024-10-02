import json


def read_json(coco_file):
    with open(coco_file, "r") as f:
        coco_data = json.load(f)
    return coco_data
