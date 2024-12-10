# dataset_generator.py

import os
import json
import random
from PIL import Image, ImageDraw
import numpy as np

# Configuration
IMAGE_SIZE = (128, 128)  # Width, Height
NUM_IMAGES = 1000
MAX_SHAPES_PER_IMAGE = 5
MIN_SHAPE_SIZE = 10
MAX_SHAPE_SIZE = 30

# Output directories
OUTPUT_DIR = 'dataset'
TRAIN_DIR = os.path.join(OUTPUT_DIR, 'train')
VAL_DIR = os.path.join(OUTPUT_DIR, 'val')
TRAIN_IMAGES_DIR = os.path.join(TRAIN_DIR, 'images')
TRAIN_LABELS_DIR = os.path.join(TRAIN_DIR, 'labels')
VAL_IMAGES_DIR = os.path.join(VAL_DIR, 'images')
VAL_LABELS_DIR = os.path.join(VAL_DIR, 'labels')

# Classes
CLASSES = ['circle', 'square']

def create_dirs():
    for directory in [TRAIN_IMAGES_DIR, TRAIN_LABELS_DIR, VAL_IMAGES_DIR, VAL_LABELS_DIR]:
        os.makedirs(directory, exist_ok=True)

def generate_shape(draw, existing_boxes):
    shape_type = random.choice(CLASSES)
    for _ in range(100):  # Try 100 times to place a non-overlapping shape
        size = random.randint(MIN_SHAPE_SIZE, MAX_SHAPE_SIZE)
        x = random.randint(0, IMAGE_SIZE[0] - size - 1)
        y = random.randint(0, IMAGE_SIZE[1] - size - 1)
        bbox = [x, y, x + size, y + size]
        
        if not check_overlap(bbox, existing_boxes):
            if shape_type == 'circle':
                radius = size // 2
                center = (x + radius, y + radius)
                draw.ellipse([center[0]-radius, center[1]-radius, center[0]+radius, center[1]+radius], fill='black')
            elif shape_type == 'square':
                draw.rectangle(bbox, fill='black')
            return shape_type, bbox
    return None, None

def check_overlap(bbox, existing_boxes, iou_threshold=0.1):
    for box in existing_boxes:
        iou = compute_iou(bbox, box)
        if iou > iou_threshold:
            return True
    return False

def compute_iou(boxA, boxB):
    # Compute Intersection over Union
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def generate_image(index, split='train'):
    img = Image.new('RGB', IMAGE_SIZE, color='white')
    draw = ImageDraw.Draw(img)
    num_shapes = random.randint(1, MAX_SHAPES_PER_IMAGE)
    objects = []
    existing_boxes = []

    for _ in range(num_shapes):
        shape_type, bbox = generate_shape(draw, existing_boxes)
        if shape_type:
            objects.append({
                'label': shape_type,
                'bbox': bbox
            })
            existing_boxes.append(bbox)

    # Save image
    if split == 'train':
        img_path = os.path.join(TRAIN_IMAGES_DIR, f'image_{index}.png')
        label_path = os.path.join(TRAIN_LABELS_DIR, f'image_{index}.json')
    else:
        img_path = os.path.join(VAL_IMAGES_DIR, f'image_{index}.png')
        label_path = os.path.join(VAL_LABELS_DIR, f'image_{index}.json')

    img.save(img_path)

    # Save labels
    with open(label_path, 'w') as f:
        json.dump({"objects": objects}, f)

def split_dataset():
    all_indices = list(range(NUM_IMAGES))
    random.shuffle(all_indices)
    split_point = int(0.8 * NUM_IMAGES)
    train_indices = all_indices[:split_point]
    val_indices = all_indices[split_point:]
    return train_indices, val_indices

def main():
    create_dirs()
    train_indices, val_indices = split_dataset()
    print(f"Generating {len(train_indices)} training images...")
    for idx in train_indices:
        generate_image(idx, split='train')
    print(f"Generating {len(val_indices)} validation images...")
    for idx in val_indices:
        generate_image(idx, split='val')
    print("Dataset generation complete.")

if __name__ == "__main__":
    main()
