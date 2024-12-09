import os
import json
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt

def generate_random_shapes_dataset(output_dir, num_images, max_shapes=5):
    """
    Generates a dataset with random shapes and their bounding boxes.

    Parameters:
        output_dir (str): Directory to save the dataset.
        num_images (int): Total number of images to generate.
        max_shapes (int): Maximum number of shapes per image.

    Returns:
        list: A list of annotations with bounding box information.
    """
    image_size = (256, 256)  # 256x256 pixels
    shapes = ["triangle", "circle", "square"]
    fill_types = [True, False]  # True for filled, False for unfilled
    os.makedirs(output_dir, exist_ok=True)

    annotations = []  # Store annotations for each image

    for img_idx in range(num_images):
        image = np.full((*image_size, 3), 255, dtype=np.uint8)  # Blank white image
        num_shapes = random.randint(1, max_shapes)  # Limit the number of shapes per image
        shape_annotations = []  # Annotations for this specific image

        for _ in range(num_shapes):
            shape_type = random.choice(shapes)
            is_filled = random.choice(fill_types)  # Randomly choose filled or unfilled
            color = tuple(map(int, np.random.randint(0, 256, size=3)))

            if shape_type == "triangle":
                pts = np.array([
                    [random.randint(30, 200), random.randint(30, 200)],
                    [random.randint(30, 200), random.randint(30, 200)],
                    [random.randint(30, 200), random.randint(30, 200)]
                ])
                pts = pts.reshape((-1, 1, 2))
                if is_filled:
                    cv2.fillPoly(image, [pts], color)
                else:
                    cv2.polylines(image, [pts], isClosed=True, color=color, thickness=2)
                bbox = [
                    int(min(pt[0][0] for pt in pts)),
                    int(min(pt[0][1] for pt in pts)),
                    int(max(pt[0][0] for pt in pts)),
                    int(max(pt[0][1] for pt in pts))
                ]
                shape_annotations.append({"shape": "triangle", "filled": is_filled, "bbox": bbox})
            elif shape_type == "circle":
                center = (random.randint(30, 200), random.randint(30, 200))
                radius = random.randint(20, 50)
                if is_filled:
                    cv2.circle(image, center, radius, color, -1)
                else:
                    cv2.circle(image, center, radius, color, 2)
                bbox = [
                    center[0] - radius,
                    center[1] - radius,
                    center[0] + radius,
                    center[1] + radius
                ]
                shape_annotations.append({"shape": "circle", "filled": is_filled, "bbox": bbox})
            elif shape_type == "square":
                top_left = (random.randint(30, 200), random.randint(30, 200))
                size = random.randint(30, 50)
                bottom_right = (top_left[0] + size, top_left[1] + size)
                if is_filled:
                    cv2.rectangle(image, top_left, bottom_right, color, -1)
                else:
                    cv2.rectangle(image, top_left, bottom_right, color, 2)
                bbox = [top_left[0], top_left[1], bottom_right[0], bottom_right[1]]
                shape_annotations.append({"shape": "square", "filled": is_filled, "bbox": bbox})

        filename = os.path.join(output_dir, f"image_{img_idx}.png")
        cv2.imwrite(filename, image)
        annotations.append({"image": filename, "shapes": shape_annotations})

    print(f"Dataset with {num_images} images (max {max_shapes} shapes per image) generated in '{output_dir}'.")
    return annotations

def split_dataset(output_dir, annotations, train_split=0.8):
    """
    Splits the dataset into training and testing sets and saves annotations.

    Parameters:
        output_dir (str): Path to the dataset directory.
        annotations (list): List of annotations.
        train_split (float): Ratio of training data.
    """
    # Create directories
    train_dir = os.path.join(output_dir, "train")
    test_dir = os.path.join(output_dir, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Clear existing files in train and test directories
    for folder in [train_dir, test_dir]:
        for file in os.listdir(folder):
            file_path = os.path.join(folder, file)
            if os.path.isfile(file_path):
                os.remove(file_path)

    # Split the dataset
    split_index = int(len(annotations) * train_split)
    train_annotations = annotations[:split_index]
    test_annotations = annotations[split_index:]

    # Move files to respective directories
    for ann in train_annotations:
        src = ann["image"]
        dst = os.path.join(train_dir, os.path.basename(src))
        os.rename(src, dst)
        ann["image"] = dst  # Update the path in annotations

    for ann in test_annotations:
        src = ann["image"]
        dst = os.path.join(test_dir, os.path.basename(src))
        os.rename(src, dst)
        ann["image"] = dst  # Update the path in annotations

    # Save annotations as JSON
    with open(os.path.join(output_dir, "train_annotations.json"), "w") as f:
        json.dump(train_annotations, f, indent=4)
    with open(os.path.join(output_dir, "test_annotations.json"), "w") as f:
        json.dump(test_annotations, f, indent=4)

    print(f"Dataset split into training and testing sets. "
          f"Training: {len(train_annotations)} images, Testing: {len(test_annotations)} images.")

def display_first_images(output_dir, num_images_to_display=10):
    """
    Displays the first few images from the dataset.

    Parameters:
        output_dir (str): Path to the dataset directory.
        num_images_to_display (int): Number of images to display.
    """
    image_paths = [os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith(".png")]
    image_paths = sorted(image_paths)[:num_images_to_display]

    plt.figure(figsize=(20, 5))
    for idx, image_path in enumerate(image_paths):
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.subplot(1, num_images_to_display, idx + 1)
        plt.imshow(img)
        plt.axis('off')
        plt.title(f"Image {idx+1}")
    plt.tight_layout()
    plt.show()

def display_images_with_correct_bounding_boxes(output_dir, num_images_to_display=10):
    """
    Displays the first few images from the dataset with bounding boxes.

    Parameters:
        output_dir (str): Path to the dataset directory.
        num_images_to_display (int): Number of images to display.
    """
    # Load annotations from the train_annotations.json file
    annotations_file = os.path.join(output_dir, "train_annotations.json")
    with open(annotations_file, "r") as f:
        annotations = json.load(f)
    
    plt.figure(figsize=(20, 10))
    for idx, ann in enumerate(annotations[:num_images_to_display]):
        img_path = ann["image"]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Draw bounding boxes on the image
        for shape in ann["shapes"]:
            if "bbox" in shape:  # Ensure bbox key exists
                bbox = shape["bbox"]
                x_min, y_min, x_max, y_max = bbox
                # Draw bounding box with a dashed black line
                step = 10  # Step size for dashed lines
                for i in range(x_min, x_max, step):  # Top and bottom edges
                    cv2.line(img, (i, y_min), (i + 5, y_min), (0, 0, 0), thickness=2)
                    cv2.line(img, (i, y_max), (i + 5, y_max), (0, 0, 0), thickness=2)
                for i in range(y_min, y_max, step):  # Left and right edges
                    cv2.line(img, (x_min, i), (x_min, i + 5), (0, 0, 0), thickness=2)
                    cv2.line(img, (x_max, i), (x_max, i + 5), (0, 0, 0), thickness=2)

        # Plot the image
        plt.subplot(2, 5, idx + 1)
        plt.imshow(img)
        plt.axis('off')
        plt.title(f"Image {idx+1}")
    
    plt.tight_layout()
    plt.show()


# Main execution
if __name__ == "__main__":
    output_directory = "shapes_dataset"
    annotations = generate_random_shapes_dataset(output_directory, 1000, max_shapes=5)
    split_dataset(output_directory, annotations)

    # Display images
    display_first_images(os.path.join(output_directory, "train"))
    display_images_with_correct_bounding_boxes(output_directory)    


