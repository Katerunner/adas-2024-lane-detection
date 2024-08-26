import numpy as np
import glob
import tqdm
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed

import os

DATASET_TRN_PATH = "datasets/cul/train"
DATASET_VAL_PATH = "datasets/cul/valid"
DATASET_TST_PATH = "datasets/cul/test"

TST_FRAC = 0.1
VAL_FRAC = 0.1
TRN_FRAC = 1 - (VAL_FRAC + TST_FRAC)

BBOX_H = 100
BBOX_W = 150

SKIP = 2

images_paths = glob.glob("data_original\\culane\\*\\*.jpg")
labels_paths = glob.glob("data_original\\culane\\*\\*.txt")

ild = {}

for image_path in images_paths:
    name = image_path.split("\\")[-1].replace(".jpg", "")
    fldr = image_path.split("\\")[-2]
    label_path = f"data_original\\culane\\{fldr}\\{name}.lines.txt"

    sample_name = name + "_" + fldr.replace(".MP4", "")

    if label_path in labels_paths:
        ild[sample_name] = (image_path, label_path)

num_trn = int(len(ild) * TRN_FRAC)
num_val = int(len(ild) * VAL_FRAC)
num_tst = int(len(ild) * TST_FRAC)

names = np.array(list(ild.keys()))
np.random.shuffle(names)

names_trn = names[:num_trn]
names_val = names[num_trn:num_trn + num_val]
names_tst = names[num_trn + num_val:num_trn + num_val + num_tst]

print(f"Number of train images: {len(names_trn)}")
print(f"Number of validation images: {len(names_val)}")
print(f"Number of test images: {len(names_tst)}")

for dataset_path in [DATASET_TRN_PATH, DATASET_VAL_PATH, DATASET_TST_PATH]:
    images_path = f"{dataset_path}/images"
    labels_path = f"{dataset_path}/labels"

    if not os.path.exists(images_path):
        os.makedirs(images_path)

    if not os.path.exists(labels_path):
        os.makedirs(labels_path)

yaml_text = """
path: cul/
train: 'train/images'
val: 'valid/images'
test: 'test/images'

# class names
names: 
  0: LL
  1: LC
  2: RC
  3: RR
""".strip()

with open("data_single_class.yaml", "w") as f:
    f.write(yaml_text)


def get_label_from_txt(_label_path, im_w, im_h, bb_w=20, bb_h=20, skip=2):
    with open(_label_path, "r") as f:
        data = [[float(j) for j in i.strip().split()] for i in f.read().split("\n")]

    label = ""
    for i in range(len(data)):
        for j in range(2, len(data[i]) - 3, 2 * skip):
            x = round(np.clip(float(data[i][j]) / im_w, 0, 1), 4)
            y = round(np.clip(float(data[i][j + 1]) / im_h, 0, 1), 4)
            w = round(np.clip(bb_w / im_w, 0, 1), 4)
            h = round(np.clip(bb_h / im_h, 0, 1), 4)

            w_offset_l = np.clip(0 - (x - w / 2), 0, 1)
            w_offset_r = np.clip((x + w / 2) - 1, 0, 1)
            h_offset_d = np.clip(0 - (y - h / 2), 0, 1)
            h_offset_u = np.clip((y + h / 2) - 1, 0, 1)

            x = x + w_offset_l - w_offset_r
            y = y + h_offset_d - h_offset_u

            label += f"{i} {x} {y} {w} {h}\n"

    return label


def _process_image_and_label(_name):
    if _name in names_trn:
        dataset_path_prefix = DATASET_TRN_PATH
    elif _name in names_val:
        dataset_path_prefix = DATASET_VAL_PATH
    else:
        dataset_path_prefix = DATASET_TST_PATH

    dataset_image_path = f"{dataset_path_prefix}/images/{_name}.png"
    dataset_label_path = f"{dataset_path_prefix}/labels/{_name}.txt"

    _image_path, _label_path = ild[_name]

    image = Image.open(_image_path)
    im_w, im_h = image.size
    image.save(dataset_image_path)
    image.close()

    label = get_label_from_txt(_label_path, im_w, im_h, bb_w=BBOX_W, bb_h=BBOX_H, skip=SKIP)

    with open(dataset_label_path, "w") as f:
        f.write(label)


with ThreadPoolExecutor(max_workers=50) as executor:
    futures = [executor.submit(_process_image_and_label, name) for name in list(ild.keys())]
    for future in tqdm.tqdm(as_completed(futures), total=len(futures), position=0):
        future.result()
