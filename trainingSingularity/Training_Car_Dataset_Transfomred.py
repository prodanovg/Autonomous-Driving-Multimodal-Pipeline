from roboflow import Roboflow
from ultralytics import YOLO

from collections import Counter

import os
import random
import shutil

ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY_ENV")

rf = Roboflow(api_key=ROBOFLOW_API_KEY)

car_project = rf.workspace("datasciencepro").project("self-driving-car-0mkga")

car_dataset = car_project.version(1).download("yolov8")

print(os.listdir(car_dataset.location))

base_path = car_dataset.location + "/export"

images_path = os.path.join(base_path, "images")
labels_path = os.path.join(base_path, "labels")

output_path = car_dataset.location + "/split"

splits = ["train", "valid", "test"]

for split in splits:
    os.makedirs(
        os.path.join(output_path, split, "images"),
        exist_ok=True
    )

    os.makedirs(
        os.path.join(output_path, split, "labels"),
        exist_ok=True
    )

images = [
    f for f in os.listdir(images_path)
    if f.endswith(".jpg")
]

random.shuffle(images)

train_split = int(0.8 * len(images))
val_split = int(0.9 * len(images))

train_files = images[:train_split]
val_files = images[train_split:val_split]
test_files = images[val_split:]

class_mapping = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 4,
    6: 5,
    7: 5,
    8: 6,
    9: 6,
    10: 7
}


def process_files(files, split):
    for file in files:

        shutil.copy(
            os.path.join(images_path, file),
            os.path.join(output_path, split, "images", file)
        )

        label_file = file.replace(".jpg", ".txt")

        src_label = os.path.join(labels_path, label_file)

        dst_label = os.path.join(
            output_path,
            split,
            "labels",
            label_file
        )

        if os.path.exists(src_label):

            updated_lines = []

            with open(src_label, "r") as f:
                lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) == 0:
                    continue
                old_class = int(parts[0])
                new_class = class_mapping[old_class]
                parts[0] = str(new_class)
                updated_lines.append(" ".join(parts))

            with open(dst_label, "w") as f:
                f.write("\n".join(updated_lines))


process_files(train_files, "train")
process_files(val_files, "valid")
process_files(test_files, "test")

print("Dataset split and class remapping complete")

data_yaml = f"""
path: {output_path}

train: train/images
val: valid/images
test: test/images

nc: 8

names: [
    "biker",
    "car",
    "pedestrian",
    "trafficLight",
    "trafficLightGreen",
    "trafficLightRed",
    "trafficLightYellow",
    "truck"
]
"""

with open("data.yaml", "w") as f:
    f.write(data_yaml)

print("data.yaml created")

with open("data.yaml", "r") as f:
    print(f.read())

labels_root = output_path

class_names = {
    0: "biker",
    1: "car",
    2: "pedestrian",
    3: "trafficLight",
    4: "green",
    5: "red",
    6: "yellow",
    7: "truck"
}

counter = Counter()

for split in ["train", "valid", "test"]:

    labels_path = os.path.join(labels_root, split, "labels")

    for file in os.listdir(labels_path):

        if file.endswith(".txt"):

            with open(os.path.join(labels_path, file), "r") as f:

                lines = f.readlines()

            for line in lines:

                parts = line.strip().split()

                if len(parts) > 0:
                    class_id = int(parts[0])

                    counter[class_id] += 1

print("Class distribution:\n")

for class_id, count in sorted(counter.items()):
    print(f"{class_names[class_id]}: {count}")


def count_images(folder):
    return len([
        f for f in os.listdir(folder)
        if f.endswith((".jpg", ".png", ".jpeg"))
    ])


train_count = count_images(
    os.path.join(output_path, "train/images")
)

val_count = count_images(
    os.path.join(output_path, "valid/images")
)

test_count = count_images(
    os.path.join(output_path, "test/images")
)

total = train_count + val_count + test_count

print(f"Train: {train_count}")
print(f"Valid: {val_count}")
print(f"Test: {test_count}")
print(f"Total: {total}")

print("\nPercentages:")

print(f"Train: {train_count / total:.2%}")
print(f"Valid: {val_count / total:.2%}")
print(f"Test: {test_count / total:.2%}")

model = YOLO("yolov8s.pt")

model.train(
    data="data.yaml",
    epochs=100,
    patience=15,
    imgsz=640,
    batch=16,
    name="car_model",
    fliplr=0.5,
    hsv_v=0.1,
    hsv_s=0.1,
    mosaic=1.0,
    close_mosaic=10,
    mixup=0.0,
    degrees=0.0
)

base_path = "runs/detect/car_model/weights"

print(os.listdir(base_path))

model = YOLO("runs/detect/car_model/weights/best.pt")

metrics = model.val(
    data="data.yaml",
    imgsz=640
)

print("mAP50:", metrics.box.map50)
print("mAP50-95:", metrics.box.map)
print("Precision:", metrics.box.mp)
print("Recall:", metrics.box.mr)


source_path = "runs/detect/car_model/weights/best.pt"
destination_path = "best.pt"

if os.path.exists(source_path):

    shutil.copy(source_path, destination_path)

    print(f"Model copied to: {destination_path}")

else:
    print("best.pt not found")

print(f"Model copied to: {destination_path}")
