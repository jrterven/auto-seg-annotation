
import os
import numpy as np
import json
import cv2
import torch


def draw_points(img, points, labels, color_obj, color_bg):
    for point, label in zip(points, labels):
        x, y = point[0], point[1]
        if label == 1:
            color = color_obj
        else:
            color = color_bg
        cv2.circle(img, (x, y), 3, color, -1)


def load_embedding(sam_predictor, embeddings_path, img_name, image, device):
    # Change extension to .pt
    filename = os.path.splitext(img_name)[0] + ".pt"
    embedding_file = os.path.join(embeddings_path, filename)

    if not os.path.exists(embedding_file):
        return False
    else:
        print(f"Embedding found in {embedding_file}")
        embedding = torch.load(embedding_file)

        sam_predictor.original_size = image.shape[:2]

        # Transform the image to the form expected by the model
        input_image = sam_predictor.transform.apply_image(image)
        input_image_torch = torch.as_tensor(input_image, device=device)
        input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]
        sam_predictor.input_size = tuple(input_image_torch.shape[-2:])
        sam_predictor.features = embedding
        sam_predictor.is_image_set = True
        return True


def save_embedding(embedding, out_path, img_name):
    # Change extension to .pt
    filename = os.path.splitext(img_name)[0] + ".pt"

    # Save embedding
    out_file = os.path.join(out_path, filename)
    torch.save(embedding, out_file)
    print(f"Saving embedding in {out_file}")


def save_masks(masks_data, out_path, img_name):
    # Change extension to png
    filename = os.path.splitext(img_name)[0] + ".png"

    if len(masks_data) > 0:
        first_mask = masks_data[0]["mask"]
        masks_canvas = np.zeros((first_mask.shape[:2]), dtype=np.uint8)

        # loop through all the masks and draw them in a single canvas using the idx as value
        for idx, masks_datum in enumerate(masks_data):
            mask_u8 = masks_datum["mask"]
            if mask_u8 is not None:
                masks_canvas[mask_u8 == 255] = idx + 1

        # Save mask
        out_file = os.path.join(out_path, filename)
        print(f"Saving Mask in {out_file}")
        cv2.imwrite(out_file, masks_canvas)


def predict_masks(predictor, points, labels, scale, fast):
    points_arr = np.array(points) / scale
    points_arr = points_arr.astype(int)

    if fast:
        mask = predictor.point_prompt(points=points_arr,
                                       pointlabel=np.array(labels))
    else:
        mask, _, _ = predictor.predict(
            point_coords=points_arr,
            point_labels=np.array(labels),
            multimask_output=False,
        )

    mask = np.squeeze(mask)  # remove leading dimension
    mask_u8 = mask.astype(np.uint8) * 255

    return mask_u8


def load_points_and_labels(dir_path, img_name, predictor, fast, scale):
    # Change extension to json
    filename = os.path.splitext(img_name)[0] + ".json"
    file_path = os.path.join(dir_path, filename)

    data = []
    if os.path.exists(file_path):
        with open(file_path, "r") as file:
            data = json.load(file)

    for dic in data:
        points = dic["sam_points"]
        labels = dic["sam_labels"]

        mask_u8 = []
        if len(points) > 0:
            mask_u8 = predict_masks(predictor, points, labels, scale, fast)
        dic["mask"] = mask_u8
    return data


def save_points_and_labels(data, out_path, img_name):
    # Change extension to json
    filename = os.path.splitext(img_name)[0] + ".json"
    out_file_path = os.path.join(out_path, filename)

    # Remove the masks so we can save in json
    data_copy = []
    for dic in data:
        data_copy.append({'sam_points': dic['sam_points'],
                          'sam_labels': dic['sam_labels'],
                          'category_id': dic["category_id"],
                          'category_name': dic["category_name"]
                         })

    with open(out_file_path, "w") as file:
        json.dump(data_copy, file)


def get_coco_image_id(images, file_name):
    """
    Retrieves the ID of an image given its file name.
    
    :param images: List of dictionaries, where each dictionary contains metadata about an image.
    :param file_name: The file name of the image for which the ID should be retrieved.
    :return: The ID of the image if found, otherwise None.
    """
    for image in images:
        if image['file_name'] == file_name:
            return image['id']
    return None


def get_next_id(list_of_dicts):
    """
    Finds the next available image ID.

    :param images: List of dictionaries, where each dictionary contains metadata about an image.
    :return: The next available image ID.
    """
    if not list_of_dicts:
        return 1
    max_id = max(d['id'] for d in list_of_dicts)
    return max_id + 1


def update_or_add_annotation(annotations, new_annotation):
    """
    Updates an existing annotation or adds a new one based on the annotation ID.
    
    :param annotations: The list of current annotations (each annotation is a dict).
    :param new_annotation: The new annotation to add or use for updating (also a dict).
    :return: A tuple containing (updated_annotations, action_taken, index_or_none)
             where updated_annotations is the list of annotations after the operation,
             action_taken is 'updated' or 'added' to indicate what was done,
             and index_or_none is the index of the updated annotation or None if added.
    """
    # Attempt to find the annotation by ID
    for index, annotation in enumerate(annotations):
        if annotation['id'] == new_annotation['id'] and annotation["image_id"] == new_annotation["image_id"]:
            # Found the annotation, update it
            annotations[index] = new_annotation
            return f'Annotation {index} Updated'

    # Annotation ID does not exist, add the new annotation
    annotations.append(new_annotation)
    return 'New Annotation'


def add_coco_annotation(annotations, file_name, category_id, mask, ann_id,
                        sam_points, sam_values):
    h, w = mask.shape[:2]

    # Get the image id
    image_id = get_coco_image_id(annotations["images"], file_name)

    # If the image is not in the annotations
    if not image_id:
        # get the next id
        image_id = get_next_id(annotations["images"])

        # prepare the image dicionary
        image_data = {
                        "file_name": file_name,
                        "height": h,
                        "width": w,
                        "id": image_id
                     }
        # and add it to the annotation
        annotations["images"].append(image_data)
        image_action = "New image added"
    image_action = None

    # Get the mask data
    mask_data = mask_to_coco_data_single_object(mask)
    new_annotation = {
                        "id": ann_id,
                        "image_id": image_id,
                        "category_id": category_id,
                        "segmentation": mask_data["segmentation"],
                        "area": mask_data["area"],
                        "bbox": mask_data["bbox"],
                        "iscrowd": 0,
                        "sam_points": sam_points,
                        "sam_values": sam_values
                     }
    ann_action = update_or_add_annotation(annotations["annotations"],
                                          new_annotation)
    return image_action, ann_action



def create_empty_coco(out_path):
    json_dict = {
            "images": [],
            "annotations": [],
            "categories": [],
        }
    
    # Writing the json_dict to a file at out_path
    with open(out_path, 'w') as outfile:
        json.dump(json_dict, outfile, indent=4)

    return json_dict
    

def mask_to_coco_data_single_object(mask):
    _, binary_mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)

    # Find contours and hierarchy
    contours, hierarchy = cv2.findContours(binary_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None  # No contours found

    segmentation = []
    if hierarchy is not None:
        # Go through each contour
        for i, contour in enumerate(contours):
            # Ensure we are only dealing with parent contours
            if hierarchy[0, i, 3] == -1:
                # Flatten the contour and add it to the segmentation
                ext_contour = contour.flatten().tolist()
                segmentation.append(ext_contour)

                # Check for child contours (holes)
                child = hierarchy[0, i, 2]
                while child != -1:
                    # Flatten the child contour and append it after the external contour
                    hole_contour = contours[child].flatten().tolist()
                    segmentation.append(hole_contour)
                    # Move to the next child
                    child = hierarchy[0, child, 0]
    else:
        # Fallback if no hierarchy information is present
        for contour in contours:
            flat_contour = contour.flatten().tolist()
            segmentation.append(flat_contour)

    # Assuming the first contour is the external one for calculating area and bbox
    area = cv2.contourArea(contours[0])
    x, y, w, h = cv2.boundingRect(contours[0])
    bbox = [x, y, w, h]

    coco_data = {
        "segmentation": [segmentation],  # Nested list to follow COCO's format
        "area": float(area),
        "bbox": bbox
    }

    return coco_data




def load_annotations(annotations_file_path):
    with open(annotations_file_path, 'r') as file:
        annotations_list = json.load(file)
    return annotations_list


def save_coco_annotation(coco_annotations, image_name, masks_data, out_path):
    
    for idx, mask_datum in enumerate(masks_data):
        # every mask datum contains
        #{"sam_points": SAM_POINTS, "sam_labels": SAM_VALUES,
        #              "mask": MASK_U8, "category_id": cat_id,
        #              "category_name": cat_name}
        if "sam_points" in mask_datum:
            sam_points = mask_datum["sam_points"]
        else:
            sam_points = []
        if "sam_labels" in mask_datum:
            sam_values = mask_datum["sam_labels"]
        else:
            sam_values = []

        img_action, ann_action = add_coco_annotation(annotations=coco_annotations,
                                                 file_name=image_name,
                                                 category_id=mask_datum["category_id"],
                                                 mask=mask_datum["mask"],
                                                 ann_id=idx,
                                                 sam_points=sam_points,
                                                 sam_values=sam_values)
        if img_action:
            print(img_action)
        print(ann_action)

    # Writing the json_dict to a file at out_path
    with open(out_path, 'w') as outfile:
        json.dump(coco_annotations, outfile, indent=4)


def load_categories(categories_file_path):
    # Load the data
    with open(categories_file_path, 'r') as file:
        categories_list = json.load(file)
    return categories_list


def rescale_points(points, scale):
    points_scaled = [[int(p[0] / scale), int(p[1] / scale)] for p in points]
    return points_scaled

def scale_points(points, scale):
    points_scaled = [[int(p[0] * scale), int(p[1] * scale)] for p in points]
    return points_scaled

def delete_last_point(points, labels):
    if points:
        points.pop()
    if labels:
        labels.pop()
    return points, labels

def load_mask_data(masks_data, index, scale):
    if len(masks_data) <= index:
        points = []
        labels = []
    else:
        points = masks_data[index]["sam_points"]
        labels = masks_data[index]["sam_labels"]
        points = scale_points(points, scale)
        
    return points, labels

# Display all the masks in blue and the current one in red
def display_all_masks(masks_data, current_index, current_mask, image, scale):
    image_with_masks = np.copy(image)

    mask_rgb = np.zeros((current_mask.shape[:2] + (3,)), dtype=np.uint8)  # Create an initial RGB image with zeros
    if len(masks_data) > 0:
        masks_canvas = np.zeros((current_mask.shape[:2]), dtype=np.uint8)
        # loop through all the masks and draw them in a single canvas
        for idx, masks_datum in enumerate(masks_data):
            if idx == current_index:
                continue
            mask_u8 = masks_datum["mask"]
            if mask_u8 is not None:
                masks_canvas = np.bitwise_or(masks_canvas, mask_u8)

        mask_rgb[:, :, 0] = masks_canvas  # Set the blue channel to the mask labels
    # draw the current mask
    mask_rgb[:, :, 2] = current_mask  # Set the red channel to the mask labels

    mask_rgb_resized = cv2.resize(mask_rgb, None, fx=scale, fy=scale)
    image_with_masks = cv2.addWeighted(image, 0.7, mask_rgb_resized, 0.3, 0)
    return image_with_masks


# Display the saved masks
def display_saved_masks(masks_data, image, scale):
    image_with_masks = np.copy(image)

    if len(masks_data) > 0:
        first_mask = masks_data[0]["mask"]
        mask_rgb = np.zeros((first_mask.shape[:2] + (3,)), dtype=np.uint8)  # Create an initial RGB image with zeros
        masks_canvas = np.zeros((first_mask.shape[:2]), dtype=np.uint8)
        # loop through all the masks and draw them in a single canvas
        for idx, masks_datum in enumerate(masks_data):
            mask_u8 = masks_datum["mask"]
            if mask_u8 is not None:
                masks_canvas = np.bitwise_or(masks_canvas, mask_u8)

        mask_rgb[:, :, 0] = masks_canvas  # Set the blue channel to the mask labels

        mask_rgb_resized = cv2.resize(mask_rgb, None, fx=scale, fy=scale)
        image_with_masks = cv2.addWeighted(image, 0.8, mask_rgb_resized, 0.2, 0)
    else:
        return image
    return image_with_masks
