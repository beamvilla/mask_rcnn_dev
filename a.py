import json

annotation_path = "./dataset/label/test/test.json"


with open(annotation_path) as annoFile:
    annotations = json.load(annoFile)

img_metadata = annotations["_via_img_metadata"]

for _, metadata in img_metadata.items():
    num_ids = []
    img_filename = metadata["filename"]
    regions = metadata["regions"]
    
    for region in regions:
        print(json.dumps(region["shape_attributes"], indent=4, ensure_ascii=False))
    break