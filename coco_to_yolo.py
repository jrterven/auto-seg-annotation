import json
import cv2
import numpy as np
import os

def save_yolo_format(image_id, image_info, bounding_boxes, save_dir):
    txt_filename = os.path.join(save_dir, f"{image_info['file_name'].split('.')[0]}.txt")
    with open(txt_filename, 'w') as f:
        for bbox in bounding_boxes:
            class_id = bbox[0]
            x_center = bbox[1] / image_info['width']
            y_center = bbox[2] / image_info['height']
            width = bbox[3] / image_info['width']
            height = bbox[4] / image_info['height']
            f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

def load_coco_annotations(annotations_path, images_dir, save_dir):
    with open(annotations_path, 'r') as file:
        coco_data = json.load(file)

    # Group annotations by image_id
    annotations_by_image = {}
    for annotation in coco_data['annotations']:
        image_id = annotation['image_id']
        if image_id not in annotations_by_image:
            annotations_by_image[image_id] = []
        annotations_by_image[image_id].append(annotation)

    # Loop through images
    for image_info in coco_data['images']:
        image_id = image_info['id']

        if image_id not in annotations_by_image:
            print(f"Image Id:{image_id}, {image_info['file_name']} does not have annotations.")
            continue  # Skip images without annotations

        print(f"Image Id:{image_id}, {image_info['file_name']} has {len(annotations_by_image[image_id])} annotations.")
        
        image_path = os.path.join(images_dir, image_info['file_name'])
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error loading image: {image_path}")
            continue

        bounding_boxes = []

        # Draw all annotations for this image
        for annotation in annotations_by_image[image_id]:
            annotation_id = annotation["id"]
            # Assuming an extra level of nesting, iterate through each set of polygons
            for polygons in annotation['segmentation']:
                for polygon in polygons:  # Now iterating over the actual polygons
                    if not isinstance(polygon, list):
                        print(f"Segmentation format error for image ID {image_id}. Is not a list. Skipping...")
                        continue

                    try:
                        # Reshape polygon to a 2D array where each row is a point
                        poly = np.array(polygon).reshape((-1, 2))
                        cv2.polylines(image, [poly.astype(np.int32)], isClosed=True, color=np.random.randint(0, 256, size=3).tolist(), thickness=2)

                        # Extract bounding box from polygon
                        x, y, w, h = cv2.boundingRect(poly.astype(np.int32))
                        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        
                        # Convert bounding box to YOLO format
                        x_center = x + w / 2
                        y_center = y + h / 2
                        bounding_boxes.append((annotation['category_id'], x_center, y_center, w, h))

                    except Exception as e:
                        print(f"Error processing segmentation for image ID {image_id}, ann ID {annotation_id}: {e}")
                        continue
        
        # Save bounding boxes in YOLO format
        save_yolo_format(image_id, image_info, bounding_boxes, save_dir)

        # Display the image with all annotations
        cv2.imshow('Image with Masks and Bounding Boxes', image)
        cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
annotations_path = "/datasets2/Dropbox/Projects/hojas/annotations.json"
images_dir = "/datasets2/Dropbox/Projects/hojas/images"
save_dir = "/datasets2/Dropbox/Projects/hojas/annotations_yolo"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
load_coco_annotations(annotations_path, images_dir, save_dir)
