import os
import json
import random
from PIL import Image, ImageDraw
import numpy as np

# Configuration
IMAGE_SIZE = (256, 256)
NUM_IMAGES = 4000
MAX_SHAPES_PER_IMAGE = 5
MIN_SHAPE_SIZE = 20
MAX_SHAPE_SIZE = 60

# Output directories
OUTPUT_DIR = 'dataset'
TRAIN_DIR = os.path.join(OUTPUT_DIR, 'train')
VAL_DIR = os.path.join(OUTPUT_DIR, 'val')
TRAIN_IMAGES_DIR = os.path.join(TRAIN_DIR, 'images')
TRAIN_LABELS_DIR = os.path.join(TRAIN_DIR, 'labels')
VAL_IMAGES_DIR = os.path.join(VAL_DIR, 'images')
VAL_LABELS_DIR = os.path.join(VAL_DIR, 'labels')

# Classes (no rectangle this time)
#CLASSES = ['circle', 'square', 'triangle']
CLASSES = ['star', 'hexagon', 'arrow']

def create_dirs():
    for directory in [TRAIN_IMAGES_DIR, TRAIN_LABELS_DIR, VAL_IMAGES_DIR, VAL_LABELS_DIR]:
        os.makedirs(directory, exist_ok=True)

def draw_star(draw, center, size, color):
    cx, cy = center
    spikes = 5
    outer_radius = size // 2
    inner_radius = int(outer_radius * 0.4)
    points = []
    angle_between_spikes = 2 * np.pi / spikes

    for i in range(spikes * 2):
        angle = i * angle_between_spikes / 2
        if i % 2 == 0:
            radius = outer_radius
        else:
            radius = inner_radius
        x = cx + int(radius * np.sin(angle))
        y = cy + int(radius * np.cos(angle))
        points.append((x, y))

    draw.polygon(points, fill=color)

def draw_hexagon(draw, center, size, color):
    cx, cy = center
    angle_offset = np.pi / 6
    sides = 6
    radius = size // 2
    points = [
        (cx + radius * np.cos(angle_offset + 2 * np.pi * i / sides), 
         cy + radius * np.sin(angle_offset + 2 * np.pi * i / sides))
        for i in range(sides)
    ]
    draw.polygon(points, fill=color)

def draw_arrow(draw, center, size, color):
    cx, cy = center
    points = [
        (cx - size // 3, cy - size // 2),  # Tail left
        (cx + size // 3, cy - size // 2),  # Tail right
        (cx + size // 3, cy),              # Middle right
        (cx + size // 2, cy),              # Tip
        (cx, cy + size // 2),              # Tip bottom
        (cx - size // 2, cy),              # Tip left
        (cx - size // 3, cy)               # Middle left
    ]
    draw.polygon(points, fill=color)

def draw_random_shape(shape_type, size):
    shape_img = Image.new('RGBA', (size, size), (255,255,255,0))
    draw = ImageDraw.Draw(shape_img)

    shape_color = (random.randint(0,255), random.randint(0,255), random.randint(0,255), 255)
    center = (size // 2, size // 2)
    #Dataset 1
    '''
    if shape_type == 'circle':
        draw.ellipse([0,0,size,size], fill=shape_color)
    elif shape_type == 'square':
        draw.rectangle([0,0,size,size], fill=shape_color)
    elif shape_type == 'triangle':
        triangle_points = [(size/2, 0), (0, size), (size, size)]
        draw.polygon(triangle_points, fill=shape_color)
    '''
    #Dataset 2
    if shape_type == 'star':
        draw_star(draw, center, size, shape_color)
    elif shape_type == 'hexagon':
        draw_hexagon(draw, center, size, shape_color)
    elif shape_type == 'arrow':
        draw_arrow(draw, center, size, shape_color)

    angle = random.uniform(0, 359)
    shape_img = shape_img.rotate(angle, expand=True)

    arr = np.array(shape_img)
    mask = arr[..., 3] > 0
    ys, xs = np.where(mask)

    if len(xs) == 0 or len(ys) == 0:
        return shape_img, (0, 0, shape_img.width - 1, shape_img.height - 1)

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    return shape_img, (x_min, y_min, x_max, y_max)

def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    if interArea == 0:
        return 0.0

    boxAArea = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
    boxBArea = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def check_overlap(bbox, existing_boxes, iou_threshold=0.1):
    for box in existing_boxes:
        iou = compute_iou(bbox, box)
        if iou > iou_threshold:
            return True
    return False

def place_shape_on_image(img, existing_boxes):
    shape_type = random.choice(CLASSES)
    for _ in range(100):
        size = random.randint(MIN_SHAPE_SIZE, MAX_SHAPE_SIZE)
        shape_img, shape_bbox = draw_random_shape(shape_type, size)
        sw, sh = shape_img.size

        max_x = IMAGE_SIZE[0] - sw
        max_y = IMAGE_SIZE[1] - sh
        if max_x < 0 or max_y < 0:
            continue
        x = random.randint(0, max_x)
        y = random.randint(0, max_y)

        shape_x_min = x + shape_bbox[0]
        shape_y_min = y + shape_bbox[1]
        shape_x_max = x + shape_bbox[2]
        shape_y_max = y + shape_bbox[3]
        candidate_bbox = [shape_x_min, shape_y_min, shape_x_max, shape_y_max]

        if not check_overlap(candidate_bbox, existing_boxes):
            img.paste(shape_img, (x,y), shape_img)
            return shape_type, candidate_bbox
    return None, None

def generate_image(index, split='train'):
    bg_color = (random.randint(200,255), random.randint(200,255), random.randint(200,255))
    img = Image.new('RGB', IMAGE_SIZE, color=bg_color)

    num_shapes = random.randint(1, MAX_SHAPES_PER_IMAGE)
    objects = []
    existing_boxes = []

    for _ in range(num_shapes):
        shape_type, bbox = place_shape_on_image(img, existing_boxes)
        if shape_type:
            bbox = [int(b) for b in bbox]
            objects.append({
                'label': shape_type,
                'bbox': bbox
            })
            existing_boxes.append(bbox)

    if split == 'train':
        img_path = os.path.join(TRAIN_IMAGES_DIR, f'image_{index}.png')
        label_path = os.path.join(TRAIN_LABELS_DIR, f'image_{index}.json')
    else:
        img_path = os.path.join(VAL_IMAGES_DIR, f'image_{index}.png')
        label_path = os.path.join(VAL_LABELS_DIR, f'image_{index}.json')

    img.save(img_path)
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
