import json
import cv2
import numpy as np
import os
import sys

def load_coco_annotations(annotations_path, images_dir):
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

        # Draw all annotations for this image
        for annotation in annotations_by_image[image_id]:
            annotation_id = annotation["id"]
            for polygon in annotation['segmentation']:  # Now iterating over the actual polygons
                if not isinstance(polygon, list):
                    print(f"Segmentation format error for image ID {image_id}. Is not a list. Skipping...")
                    continue

                try:
                    # Reshape polygon to a 2D array where each row is a point
                    poly = np.array(polygon).reshape((-1, 2))
                    cv2.polylines(image, [poly.astype(np.int32)], isClosed=True, color=np.random.randint(0, 256, size=3).tolist(), thickness=2)
                    
                    # Extract bounding box from polygon
                    #x, y, w, h = cv2.boundingRect(poly.astype(np.int32))
                    #cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    
                except Exception as e:
                    print(f"Error processing segmentation for image ID {image_id}, ann ID {annotation_id}: {e}")
                    continue

        # Display the image with all annotations
        cv2.imshow('Image with Masks', image)
        key = cv2.waitKey(0)
        if key == ord('q'):
            break
    cv2.destroyAllWindows()


# Example usage
#annotations_path = "seg_sample_project/annotations.json"
#images_dir = "seg_sample_project/images"

annotations_path = "/datasets2/Dropbox/AiFi_work/Hotdog/ann_project/auto_ann_project_smooth/annotations.json"
images_dir = "/datasets2/Dropbox/AiFi_work/Hotdog/ann_project/auto_ann_project_smooth/images"
load_coco_annotations(annotations_path, images_dir)
